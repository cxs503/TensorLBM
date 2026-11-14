"""Round-3 XVAL driver: D3Q19 whole-step pointwise regression vs lbmpy.

Reference step:  S_ref = definition_stream3d o Collide_lbmpy(symbolic, discrete eq)
Arms under test: every D3Q19 collide+stream composition the repo exposes.

Outputs (all numbers in the report are generated from these, never typed):
    summary.json          aggregated results + criteria verdicts + provenance
    table.csv             one row per (arm, grid, tau, seed, step)
    runs/raw_points.json  raw per-point metrics
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XV = HERE.parent
REPO = XV.parents[1]
for _p in (str(HERE), str(REPO / "src"), str(XV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms3d  # noqa: E402
import ref3d  # noqa: E402
import refcore3d as core  # noqa: E402
from lbmpy_pointwise_ref import make_test_field  # noqa: E402

PRE = json.loads((HERE / "predeclared.json").read_text())
GRIDS = [tuple(g) for g in PRE["scope"]["grids"]]
TAUS = PRE["scope"]["taus"]
STEPS = PRE["scope"]["steps"]
REF_STEPS = 30  # pointwise lbmpy cross-check window inside the long physics runs
SEEDS = PRE["scope"]["seeds"]

_REF_CACHE: dict[tuple, object] = {}


def _srt_ref(tau: float, equilibrium: str = "discrete"):
    key = ("srt", round(float(tau), 12), equilibrium)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = ref3d.reference_collider(tau, equilibrium=equilibrium, model="srt")
    return _REF_CACHE[key]


def _mrt_ref(tau: float, uniform: bool = False):
    key = ("mrt_uniform" if uniform else "mrt_native", round(float(tau), 12))
    if key not in _REF_CACHE:
        matrix, rates = ref3d.repo_mrt_rates(tau, uniform=uniform)
        _REF_CACHE[key] = ref3d.reference_collider(
            tau, model="mrt_uniform" if uniform else "mrt_from_matrix",
            moment_matrix=matrix, relaxation_rates=rates,
        )
        _REF_CACHE[("mrt_gap", round(float(tau), 12))] = ref3d.moment_basis_gap(matrix)
    return _REF_CACHE[key]


def reference_trajectory(f0, tau, steps, *, equilibrium="discrete", model="srt", keep=(1,)):
    if model == "mrt_native":
        collider = _mrt_ref(tau)
    elif model == "mrt_uniform":
        collider = _mrt_ref(tau, uniform=True)
    else:
        collider = _srt_ref(tau, equilibrium=equilibrium)
    f = np.array(f0, dtype=np.float64, copy=True)
    states = {0: f}
    want = set(keep) | {steps}
    for t in range(1, steps + 1):
        f = core.definition_stream3d(collider.apply(f))
        if t in want:
            states[t] = f
    return collider, states


# ---------------------------------------------------------------------------
# 1. code-equivalence sweep
# ---------------------------------------------------------------------------
def sweep_equivalence(rows, dense_curves, findings):
    results = []
    for grid in GRIDS:
        for tau in TAUS:
            for seed in SEEDS:
                f0 = make_test_field("D3Q19", tau=tau, seed=seed, shape=grid)
                refs = {}
                trajectories = {}
                for arm in arms3d.ARMS:
                    trajectories[arm] = arms3d.run_arm(arm, f0, tau, STEPS)
                dense = (list(grid) == [8, 8, 8] and tau == 0.7 and seed == 0)
                traj_eps: dict[str, list[dict]] = {}
                for arm, traj in trajectories.items():
                    model = {"A5_mrt_native": "mrt_native"}.get(arm, "srt")
                    if arm == "A4_mrt_uniform":
                        # the symbolic uniform-rate MRT reference is built once (tau=0.7);
                        # at other taus the arm is compared with the SRT reference, which
                        # is exactly the MRT-vs-BGK gap pinned by finding F1.
                        model = "mrt_uniform" if tau == 0.7 else "srt"
                    if model not in refs:
                        keep = tuple(range(1, STEPS + 1)) if dense else (1,)
                        refs[model] = reference_trajectory(
                            f0, tau, STEPS, model=model, keep=keep
                        )[1]
                    ref = refs[model][STEPS]
                    if dense:
                        traj_eps[arm] = [
                            core.compare(got, refs[model][t]) for t, got in enumerate(traj, start=1)
                        ]
                    cmp30 = core.compare(traj[-1], ref)
                    cmp1 = core.compare(traj[0], refs[model][1])
                    # finding F1: how far the uniform-rate MRT step is from BGK
                    if arm == "A4_mrt_uniform":
                        if "srt" not in refs:
                            refs["srt"] = reference_trajectory(
                                f0, tau, STEPS, model="srt", keep=(1,)
                            )[1]
                        gap = core.compare(traj[-1], refs["srt"][STEPS])["eps_inf"]
                        gap1 = core.compare(traj[0], refs["srt"][1])["eps_inf"]
                        findings.append(
                            {"kind": "F1_mrt_vs_bgk_gap", "grid": list(grid), "tau": tau,
                             "seed": seed, "eps_inf_t1": gap1, "eps_inf_t30": gap}
                        )
                    cons = max(
                        core.conservation(traj[i - 1] if i else f0, traj[i])["mass_drift"]
                        for i in range(len(traj))
                    )
                    cons_mom = max(
                        core.conservation(traj[i - 1] if i else f0, traj[i])["momentum_drift"]
                        for i in range(len(traj))
                    )
                    rec = {
                        "kind": "code_equivalence",
                        "arm": arm,
                        "reference": model + ("/discrete" if model == "srt" else ""),
                        "grid": list(grid),
                        "tau": tau,
                        "seed": seed,
                        "eps_inf_t1": cmp1["eps_inf"],
                        "eps_inf_t30": cmp30["eps_inf"],
                        "eps_l2_t30": cmp30["eps_l2"],
                        "mass_drift_max": cons,
                        "momentum_drift_max": cons_mom,
                    }
                    results.append(rec)
                    if dense:
                        dense_curves.append(
                            {"arm": arm, "eps_inf": [c["eps_inf"] for c in traj_eps[arm]]}
                        )
                    rows.append(
                        [
                            "code_equivalence",
                            arm,
                            "x".join(map(str, grid)),
                            tau,
                            seed,
                            STEPS,
                            f"{cmp30['eps_inf']:.6e}",
                            f"{cmp30['eps_l2']:.6e}",
                        ]
                    )
                    print(
                        f"[equiv] {arm:20s} grid={grid} tau={tau} seed={seed} "
                        f"eps_inf(t=30)={cmp30['eps_inf']:.3e} mass={cons:.1e}"
                    )
    return results


# ---------------------------------------------------------------------------
# 2. convention arm (continuous vs discrete equilibrium)
# ---------------------------------------------------------------------------
def convention_arm(rows):
    grid = (8, 8, 8)
    tau = 0.7
    f0 = make_test_field("D3Q19", tau=tau, seed=0, shape=grid, amplitude=0.15)
    arm_traj = arms3d.run_arm("A1_solver3d_bgk", f0, tau, STEPS)
    out = {}
    for eq in ("discrete", "continuous"):
        collider, refs = reference_trajectory(f0, tau, STEPS, equilibrium=eq, keep=(1,))
        out[eq] = {
            "eps_inf_t1": core.compare(arm_traj[0], refs[1])["eps_inf"],
            "eps_inf_t5": core.compare(arm_traj[4], refs[5])["eps_inf"]
            if 5 in refs
            else None,
            "eps_inf_t30": core.compare(arm_traj[-1], refs[STEPS])["eps_inf"],
            "trajectory": [
                core.compare(arm_traj[t - 1], refs[t])["eps_inf"] if t in refs else None
                for t in (1, STEPS)
            ],
        }
        for t, v in (("t1", out[eq]["eps_inf_t1"]), (("t30"), out[eq]["eps_inf_t30"])):
            if v is not None:
                rows.append(["convention", f"A1_vs_{eq}_eq", "8x8x8", tau, 0, t, f"{v:.6e}", ""])
    # rest-state mass of the continuous equilibrium on the D3Q19 stencil
    rho = np.ones(grid)
    u = np.zeros((3,) + grid)
    cont = core.continuous_equilibrium(rho, u)
    out["continuous_eq_rest_mass_ratio"] = float(cont.sum() / rho.sum())
    out["continuous_eq_rest_eps_vs_discrete"] = core.compare(
        cont, core.discrete_equilibrium(rho, u)
    )["eps_inf"]
    print(f"[convention] continuous rest-mass ratio = {out['continuous_eq_rest_mass_ratio']:.6f}")
    return out


# ---------------------------------------------------------------------------
# 3. physics arm: shear wave
# ---------------------------------------------------------------------------
def _fit_decay(times, amps, start_frac: float = 0.25):
    t = np.asarray(times, dtype=np.float64)
    a = np.asarray(amps, dtype=np.float64)
    keep = (a > 1e-8) & (t >= start_frac * t[-1])
    slope, _intercept = np.polyfit(t[keep], np.log(a[keep]), 1)
    return -float(slope), int(keep.sum())


def physics_shear_wave(rows):
    out = []
    tau_list = [0.6, 0.8, 1.0]
    for n in (16, 32):
        k = 2.0 * math.pi / n
        k_eff2 = (2.0 * math.sin(k / 2.0)) ** 2
        for tau in tau_list:
            nu = (tau - 0.5) / 3.0
            rate_cont = nu * k**2
            steps = int(math.ceil(3.0 / rate_cont))
            f = core.shear_wave_3d((n, n, n), 0.02, k)
            f0 = f
            amps = [core.shear_amplitude(f)]
            ref_collider = _srt_ref(tau)
            ref = np.array(f0, copy=True)
            ref_steps = steps  # numpy reference is cheap; compare every step
            eps = float("nan")
            for t in range(1, steps + 1):
                f = arms3d.step_solver3d_bgk(f, tau)
                if t <= ref_steps:
                    ref = core.definition_stream3d(ref_collider.apply(ref))
                amps.append(core.shear_amplitude(f))
            if ref_steps == steps:
                eps = core.compare(f, ref)["eps_inf"]
            measured, npts = _fit_decay(list(range(steps + 1)), amps)
            rec = {
                "kind": "physics_shear_wave",
                "n": n,
                "k": k,
                "k_eff_sq": k_eff2,
                "tau": tau,
                "nu": nu,
                "steps": steps,
                "fit_points": npts,
                "rate_measured": measured,
                "rate_continuum": rate_cont,
                "rate_lattice_dispersion": nu * k_eff2,
                "err_vs_continuum": abs(measured - rate_cont) / rate_cont,
                "err_vs_lattice_dispersion": abs(measured - nu * k_eff2) / (nu * k_eff2),
                "eps_inf_vs_lbmpy_ref": eps,
                "amplitude_final": amps[-1],
            }
            out.append(rec)
            rows.append(
                [
                    "physics_shear_wave",
                    "A1_solver3d_bgk",
                    f"{n}x{n}x{n}",
                    tau,
                    0,
                    steps,
                    f"{rec['err_vs_continuum']:.6e}",
                    f"{rec['err_vs_lattice_dispersion']:.6e}",
                ]
            )
            print(
                f"[shear] n={n} tau={tau} rate={measured:.6e} "
                f"err_cont={rec['err_vs_continuum']:.2%} err_keff={rec['err_vs_lattice_dispersion']:.2%} "
                f"eps_ref={eps:.1e}"
            )
    return out


# ---------------------------------------------------------------------------
# 4. physics arm: ABC (Beltrami) flow, exact decay exp(-nu t)
# ---------------------------------------------------------------------------
def physics_abc(rows):
    out = []
    for n in (16,):
        for tau in [0.6, 0.8]:
            nu = (tau - 0.5) / 3.0
            steps = min(int(math.ceil(3.0 / nu)), 300)
            f = core.abc_field(n, 0.02)
            f0 = f
            amps = [core.abc_amplitude(f)]
            ref_collider = _srt_ref(tau)
            ref = np.array(f0, copy=True)
            eps = float("nan")
            for t in range(1, steps + 1):
                f = arms3d.step_solver3d_bgk(f, tau)
                ref = core.definition_stream3d(ref_collider.apply(ref))
                amps.append(core.abc_amplitude(f))
            eps = core.compare(f, ref)["eps_inf"]
            measured, npts = _fit_decay(list(range(steps + 1)), amps)
            rec = {
                "kind": "physics_abc",
                "n": n,
                "tau": tau,
                "nu": nu,
                "steps": steps,
                "fit_points": npts,
                "rate_measured": measured,
                "rate_analytic": nu,
                "err_vs_analytic": abs(measured - nu) / nu,
                "eps_inf_vs_lbmpy_ref": eps,
            }
            out.append(rec)
            rows.append(
                [
                    "physics_abc",
                    "A1_solver3d_bgk",
                    f"{n}x{n}x{n}",
                    tau,
                    0,
                    steps,
                    f"{rec['err_vs_analytic']:.6e}",
                    "",
                ]
            )
            print(
                f"[abc] n={n} tau={tau} rate={measured:.6e} nu={nu:.6e} "
                f"err={rec['err_vs_analytic']:.2%} eps_ref={eps:.1e}"
            )
    return out


# ---------------------------------------------------------------------------
# 5. sensitivity controls
# ---------------------------------------------------------------------------
def sensitivity_controls():
    grid = (8, 8, 8)
    tau = 0.7
    f0 = make_test_field("D3Q19", tau=tau, seed=0, shape=grid)
    arm = arms3d.run_arm("A1_solver3d_bgk", f0, tau, STEPS)
    ref_true = reference_trajectory(f0, tau, STEPS, keep=(1,))[1][STEPS]
    # control 1: the ARM relaxation time scaled by 1.001 (no extra symbolic build)
    arm_bad = arms3d.run_arm("A1_solver3d_bgk", f0, tau * 1.001, STEPS)
    eps_tau = core.compare(arm_bad[-1], ref_true)["eps_inf"]
    # control 2: reference collision applied with the direction permutation dropped
    collider = _srt_ref(tau)
    perm = collider.perm.copy()
    collider.perm = np.arange(len(perm))
    f = np.array(f0, copy=True)
    for _ in range(STEPS):
        f = core.definition_stream3d(collider.apply(f))
    collider.perm = perm  # restore
    eps_perm = core.compare(arm[-1], f)["eps_inf"]
    print(f"[control] tau*1.001 -> eps={eps_tau:.2e} ; no-permutation -> eps={eps_perm:.2e}")
    return {
        "tau_perturbation_1.001": {
            "eps_inf": eps_tau,
            "detected": bool(eps_tau > PRE["acceptance_criteria"]["C8_sensitivity_controls"]
                             ["tau_perturbation"]["must_detect_above"]),
        },
        "permutation_dropped": {
            "eps_inf": eps_perm,
            "detected": bool(eps_perm > PRE["acceptance_criteria"]["C8_sensitivity_controls"]
                             ["permutation_control"]["must_detect_above"]),
        },
    }


# ---------------------------------------------------------------------------
# 6. frozen reference bundle (lbmpy practice: stored reference values)
# ---------------------------------------------------------------------------
FREEZE = {
    "grid": (6, 9, 13),
    "tau": 0.7,
    "seed": 0,
    "steps": STEPS,
}


def freeze_bundle() -> dict:
    """Write refdata/pointwise_d3q19_v1.npz + manifest with per-array sha256.

    The bundle stores (a) the reference side produced by lbmpy+definition
    streaming and (b) the current repo outputs, so the regression gate can run
    on a machine without lbmpy and still detect an unintended change on either
    side.
    """
    import hashlib

    grid = FREEZE["grid"]
    tau = FREEZE["tau"]
    f0 = make_test_field("D3Q19", tau=tau, seed=FREEZE["seed"], shape=grid)
    _, refs = reference_trajectory(f0, tau, FREEZE["steps"], keep=(1, FREEZE["steps"]))
    _, refs_cont = reference_trajectory(
        f0, tau, FREEZE["steps"], equilibrium="continuous", keep=(1,)
    )
    arrays = {
        "f0": f0,
        "ref_t1": refs[1],
        "ref_t30": refs[FREEZE["steps"]],
        "ref_continuous_t1": refs_cont[1],
    }
    for arm in arms3d.ARMS:
        arrays[f"repo_{arm}_t30"] = arms3d.run_arm(arm, f0, tau, FREEZE["steps"])[-1]
    out_dir = HERE / "refdata"
    out_dir.mkdir(exist_ok=True)
    npz_path = out_dir / "pointwise_d3q19_v1.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest = {
        "file": npz_path.name,
        "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "arrays": {k: hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest()
                   for k, v in arrays.items()},
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "config": {**FREEZE, "grid": list(grid)},
        "provenance": ref3d.provenance(),
    }
    (out_dir / "pointwise_d3q19_v1.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[freeze] {npz_path.name} sha256={manifest['sha256'][:16]} arrays={len(arrays)}")
    return {"npz": str(npz_path.relative_to(REPO)), "sha256": manifest["sha256"],
            "n_arrays": len(arrays)}


def _dump(summary: dict) -> None:
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


def main() -> int:
    t0 = time.time()
    rows: list[list] = []
    summary: dict = {"anchor": PRE["anchor"], "round": 3, "lattice": "D3Q19"}

    summary["provenance"] = ref3d.provenance()
    summary["predeclared"] = PRE["scope"]

    dense_curves: list[dict] = []
    findings: list[dict] = []
    equiv = sweep_equivalence(rows, dense_curves, findings)
    summary["code_equivalence"] = equiv
    summary["per_step_curves"] = dense_curves
    summary["findings"] = findings

    _dump(summary)
    summary["convention"] = convention_arm(rows)
    _dump(summary)
    summary["frozen_bundle"] = freeze_bundle()
    _dump(summary)
    summary["physics_shear_wave"] = physics_shear_wave(rows)
    _dump(summary)
    summary["physics_abc"] = physics_abc(rows)
    _dump(summary)
    summary["sensitivity"] = sensitivity_controls()
    _dump(summary)

    # ---- criteria verdicts ------------------------------------------------
    c1 = PRE["acceptance_criteria"]["C1_code_equivalence"]["threshold"]
    c2 = PRE["acceptance_criteria"]["C2_mrt_native_vs_symbolic"]["threshold"]
    c3 = PRE["acceptance_criteria"]["C3_mrt_basis_reconstruction"]["threshold"]
    c4 = PRE["acceptance_criteria"]["C4_conservation"]["threshold"]
    c6 = PRE["acceptance_criteria"]["C6_physics_shear_wave"]["threshold"]
    c7 = PRE["acceptance_criteria"]["C7_physics_abc"]["threshold"]

    worst = {}
    for rec in equiv:
        arm = rec["arm"]
        worst[arm] = max(worst.get(arm, 0.0), rec["eps_inf_t30"])
    summary["worst_eps_inf_t30_by_arm"] = worst

    verdicts = {
        "C1_code_equivalence": {
            "threshold": c1,
            "worst": max(worst.get(a, 0.0) for a in ("A1_solver3d_bgk", "A2_diff3d_step", "A3_adv_bgk_d3q19")),
            "passed": all(worst.get(a, 1.0) <= c1
                          for a in ("A1_solver3d_bgk", "A2_diff3d_step", "A3_adv_bgk_d3q19")),
        },
        "C2b_mrt_uniform_vs_symbolic": {
            "threshold": c2,
            "worst": max([r["eps_inf_t30"] for r in equiv
                          if r["reference"].startswith("mrt_uniform")] or [float("nan")]),
            "passed": bool(max([r["eps_inf_t30"] for r in equiv
                                if r["reference"].startswith("mrt_uniform")] or [1.0]) <= c2),
        },
        "C2_mrt_native_vs_symbolic": {
            "threshold": c2,
            "worst": worst.get("A5_mrt_native"),
            "passed": bool(worst.get("A5_mrt_native", 1.0) <= c2),
        },
        "C3_mrt_basis_reconstruction": {
            "threshold": c3,
            "value": max(
                [v for k, v in _REF_CACHE.items() if isinstance(k, tuple) and k[0] == "mrt_gap"]
                or [float("nan")]
            ),
        },
        "C4_conservation": {
            "threshold": c4,
            "worst_mass": max(r["mass_drift_max"] for r in equiv),
            "worst_momentum": max(r["momentum_drift_max"] for r in equiv),
        },
        "C6_physics_shear_wave": {
            "threshold": c6,
            "worst_err_vs_continuum": max(r["err_vs_continuum"] for r in summary["physics_shear_wave"]),
            "worst_err_vs_lattice_dispersion": max(
                r["err_vs_lattice_dispersion"] for r in summary["physics_shear_wave"]
            ),
        },
        "C7_physics_abc": {
            "threshold": c7,
            "worst_err": max(r["err_vs_analytic"] for r in summary["physics_abc"]),
            "finest_grid_err": [
                r["err_vs_analytic"] for r in summary["physics_abc"] if r["n"] == 24
            ],
        },
    }
    verdicts["C3_mrt_basis_reconstruction"]["passed"] = bool(
        verdicts["C3_mrt_basis_reconstruction"]["value"] <= c3
    )
    verdicts["C4_conservation"]["passed"] = bool(
        verdicts["C4_conservation"]["worst_mass"] <= c4
        and verdicts["C4_conservation"]["worst_momentum"] <= c4
    )
    verdicts["C6_physics_shear_wave"]["passed"] = bool(
        verdicts["C6_physics_shear_wave"]["worst_err_vs_lattice_dispersion"] <= c6
    )
    verdicts["C7_physics_abc"]["passed"] = bool(
        verdicts["C7_physics_abc"]["worst_err"] <= c7
        or (verdicts["C7_physics_abc"]["finest_grid_err"]
            and verdicts["C7_physics_abc"]["finest_grid_err"][0] <= c7)
    )
    verdicts["C8_sensitivity_controls"] = {
        "passed": all(v["detected"] for v in summary["sensitivity"].values()),
        **summary["sensitivity"],
    }
    summary["verdicts"] = verdicts
    summary["runtime_s"] = round(time.time() - t0, 2)

    (HERE / "runs").mkdir(exist_ok=True)
    (HERE / "runs" / "raw_points.json").write_text(json.dumps(equiv, indent=1))
    with (HERE / "table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "arm_or_side", "grid", "tau", "seed", "steps", "metric_1", "metric_2"])
        w.writerows(rows)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(verdicts, indent=1, default=float))
    print(f"done in {summary['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())