"""Round 2 of XVAL-LBM-POINTWISE-REFERENCE-REGRESSION: full-step pointwise gate.

Arms
----
S   full-step (collide + stream) pointwise trajectory vs the reference
    ``definition_stream(lbmpy_collision(f))``, 30 steps.
X   cross-implementation audit: every in-repo streaming kernel of the repo
    checked pointwise against the definition (each in its native layout).
P   physics arm: shear-wave amplitude decay vs ``exp(-nu k^2 t)``.

Outputs (``xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/stepwise/``):
    summary.json, table.csv, runs/raw.npz
Run ``make_figures.py`` afterwards for the figures used by the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepwise_ref as S  # noqa: E402

OUT = HERE
RUNS = OUT / "runs"
GATES = json.loads((OUT / "predeclared.json").read_text())["gates"]


def _iso(i: int) -> str:
    return str(i).zfill(2)


# ---------------------------------------------------------------------------
# S arms
# ---------------------------------------------------------------------------
def shear_field(spec: dict) -> np.ndarray:
    kind = spec["field"]
    if kind.startswith("shear_wave"):
        return S.shear_wave_field(spec.get("ly", 64), spec.get("lx", 8), u0=spec.get("u0", 2e-2))
    if kind.startswith("non-square"):
        rng = np.random.default_rng(7)
        return (S.shear_wave_field(7, 11, u0=5e-2)
                + 1e-3 * rng.standard_normal((9, 7, 11)))
    if kind.startswith("high-Ma"):
        rng = np.random.default_rng(11)
        return (S.shear_wave_field(32, 32, u0=1e-1)
                + 1e-2 * rng.standard_normal((9, 32, 32)))
    raise ValueError(kind)


def _params(case: dict) -> dict:
    p = {"tau": case["tau"]}
    if case["model"] == "trt":
        p["lambda_trt"] = case.get("lambda_trt", 3.0 / 16.0)
    return p


def run_stepwise(case: dict) -> dict:
    n = case["steps"]
    f0 = shear_field(case)
    ref = S.StepwiseReference(lattice="D2Q9", model=case["model"], **_params(case))
    f_ref = f0.copy()
    f_tl = f0.copy()
    mass0 = float(f0.sum())
    mom0 = np.tensordot(S.ref.tensorlbm_velocities("D2Q9"), f0, axes=(0, 0))
    traj = []
    for t in range(1, n + 1):
        f_ref = ref.step(f_ref)
        f_tl = S.in_repo_step(f_tl, model=case["model"], **_params(case))
        if t in (1, 5, 15, n):
            c = S.compare(f_tl, f_ref)
            traj.append({"step": t, **c})
    c_final = traj[-1]
    mass_drift = abs(float(f_tl.sum()) - mass0) / abs(mass0)
    # Total momentum of a periodic box is exactly conserved; the *relative*
    # drift against |sum j| is meaningless here (a shear wave has sum j = 0),
    # so normalise the absolute drift by the total mass (per-node momentum).
    mom = np.tensordot(S.tensorlbm_velocities("D2Q9"), f_tl, axes=(0, 0))
    tot0 = mom0.sum(axis=(1, 2))
    mom_drift = float(np.abs(mom.sum(axis=(1, 2)) - tot0).max() / abs(mass0))
    return {
        "arm": case["arm"], "model": case["model"], "tau": case["tau"],
        "lambda_trt": case.get("lambda_trt"), "field": case["field"], "steps": n,
        "eps_scaled_inf": c_final["eps_scaled_inf"],
        "eps_rel_rms": c_final["eps_rel_rms"],
        "eps_scaled_inf_max_over_traj": max(r["eps_scaled_inf"] for r in traj),
        "gate": GATES["G_code_equivalence"]["value"],
        "pass": c_final["eps_scaled_inf"] < GATES["G_code_equivalence"]["value"],
        "mass_drift_rel": mass_drift,
        "momentum_drift_abs_over_mass": mom_drift,
        "conservation_pass": (mass_drift < GATES["G_conservation"]["value"]
                              and mom_drift < GATES["G_conservation"]["value"]),
        "trajectory": traj,
        "_diff_field": (f_tl - f_ref) / max(float(np.abs(f_ref).max()), 1e-30),
        "_f_ref_final": f_ref,
    }


# ---------------------------------------------------------------------------
# X arms: streaming cross-check in every native layout
# ---------------------------------------------------------------------------
def run_streaming_audit() -> tuple[list[dict], dict[str, np.ndarray]]:
    import torch

    rng = np.random.default_rng(0)
    ny, nx = 7, 11
    f2 = 0.2 + rng.standard_normal((9, ny, nx))
    fdef2 = S.definition_stream(f2)

    rows: list[dict] = []
    keep: dict[str, np.ndarray] = {}

    def add(name, layout, got, fdef, note=""):
        c = S.compare(np.asarray(got, dtype=np.float64), fdef)
        rows.append({"impl": name, "layout": layout, "lattice": "D2Q9",
                     "eps_scaled_inf": c["eps_scaled_inf"],
                     "gate": GATES["G_stream_crosscheck"]["value"],
                     "pass": c["eps_scaled_inf"] < GATES["G_stream_crosscheck"]["value"],
                     "note": note})
        keep[name] = np.asarray(got, dtype=np.float64)

    from tensorlbm import diff_boundary_grad, lattice, les_bptt, solver

    add("solver.stream", "(9,ny,nx) gather",
        solver.stream(torch.from_numpy(f2.copy())).numpy(), fdef2)
    add("lattice.stream", "(ny,nx,9) per-direction roll",
        np.moveaxis(lattice.stream(torch.from_numpy(np.moveaxis(f2, 0, -1).copy())).numpy(), -1, 0),
        fdef2)
    add("les_bptt.stream", "(B,9,ny,nx) conv2d",
        les_bptt.stream(torch.from_numpy(f2[None].copy())).numpy()[0], fdef2,
        note="BPTT kernel, depthwise conv instead of roll")
    add("diff_boundary_grad.stream", "(9,ny,nx)",
        diff_boundary_grad.stream(torch.from_numpy(f2.copy())).numpy(), fdef2)

    # h_theorem.stream needs an explicit LatticeSpec
    try:
        import tensorlbm.h_theorem as hth
        from tensorlbm.d2q9 import C as C2, W_EXACT

        spec = hth.LatticeSpec("D2Q9", 9, 2, C2.clone(), torch.tensor(np.array(W_EXACT)))
        add("h_theorem.stream", "(9,ny,nx) spec-driven",
            hth.stream(torch.from_numpy(f2.copy()), spec).numpy(), fdef2)
    except Exception as exc:  # noqa: BLE001 - recorded honestly, not hidden
        rows.append({"impl": "h_theorem.stream", "layout": "(9,ny,nx) spec-driven",
                     "lattice": "D2Q9", "eps_scaled_inf": float("nan"),
                     "gate": GATES["G_stream_crosscheck"]["value"], "pass": None,
                     "note": f"not run: {type(exc).__name__}: {exc}"})

    # D3Q19 (channel_kernel_numpy), native layout (19,nz,ny,nx)
    from tensorlbm import channel_kernel_numpy as ckn
    import lbmpy_pointwise_ref as ref

    nz, nyz, nxz = 5, 6, 7
    f3 = 0.2 + rng.standard_normal((19, nz, nyz, nxz))
    vel = ref.tensorlbm_velocities("D3Q19").astype(int)
    fdef3 = np.empty_like(f3)
    for q in range(19):
        fdef3[q] = np.roll(f3[q], shift=tuple(int(v) for v in vel[q]), axis=(0, 1, 2))
    c = S.compare(np.asarray(ckn.stream(f3), dtype=np.float64), fdef3)
    rows.append({"impl": "channel_kernel_numpy.stream", "layout": "(19,nz,ny,nx)",
                 "lattice": "D3Q19", "eps_scaled_inf": c["eps_scaled_inf"],
                 "gate": GATES["G_stream_crosscheck"]["value"],
                 "pass": c["eps_scaled_inf"] < GATES["G_stream_crosscheck"]["value"],
                 "note": "_shift per direction, cubic domain 5x6x7"})
    return rows, keep


# ---------------------------------------------------------------------------
# P arms: shear-wave decay vs analytic
# ---------------------------------------------------------------------------
def run_physics(case: dict) -> dict:
    n = case["steps"]
    f0 = shear_field(case)
    tau = case["tau"]
    nu = (tau - 0.5) / 3.0
    ly = f0.shape[1]
    k = 2.0 * np.pi / ly
    a0 = S.shear_amplitude(f0)
    ref = S.StepwiseReference(lattice="D2Q9", model="srt", tau=tau)
    f_ref = f0.copy()
    f_tl = f0.copy()
    ts, a_tl, a_ref = [], [], []
    for t in range(1, n + 1):
        f_tl = S.in_repo_step(f_tl, model="srt", tau=tau)
        f_ref = ref.step(f_ref)
        if t % 100 == 0:
            ts.append(t)
            a_tl.append(S.shear_amplitude(f_tl))
            a_ref.append(S.shear_amplitude(f_ref))
    a_tl = np.array(a_tl)
    a_ref = np.array(a_ref)
    ts = np.array(ts, dtype=float)
    analytic = a0 * np.exp(-nu * k ** 2 * ts)
    rel = np.abs(a_tl / analytic - 1.0)
    # log-linear fit of the measured decay rate (in-repo trajectory)
    slope = np.polyfit(ts, np.log(a_tl), 1)[0]
    return {
        "arm": case["arm"], "tau": tau, "nu": nu, "k": k, "nu_k2": nu * k ** 2,
        "steps": n, "a0": a0,
        "A_final_inrepo": float(a_tl[-1]), "A_final_reference": float(a_ref[-1]),
        "A_final_analytic": float(analytic[-1]),
        "err_vs_analytic_max_rel": float(rel.max()),
        "err_final_vs_analytic_rel": float(abs(a_tl[-1] / analytic[-1] - 1.0)),
        "fitted_rate": float(-slope), "analytic_rate": float(nu * k ** 2),
        "rate_err_rel": float(abs(-slope / (nu * k ** 2) - 1.0)),
        "eps_inrepo_vs_ref_final": S.compare(f_tl, f_ref)["eps_scaled_inf"],
        "gate": GATES["G_physics_shear_decay"]["value"],
        "pass": bool(rel.max() < GATES["G_physics_shear_decay"]["value"]),
        "series": {"t": ts.tolist(), "A_inrepo": a_tl.tolist(),
                   "A_reference": a_ref.tolist(), "A_analytic": analytic.tolist()},
    }


# ---------------------------------------------------------------------------
def main() -> int:
    dec = json.loads((OUT / "predeclared.json").read_text())
    RUNS.mkdir(exist_ok=True)
    summary: dict = {"anchor": dec["anchor"], "round": 2,
                     "gates": dec["gates"], "provenance": S.provenance(),
                     "stepwise": [], "streaming_audit": [], "physics": []}
    keep = {}
    for case in dec["cases"]:
        if case["arm"].startswith("S"):
            r = run_stepwise(case)
            keep[f"diff_{case['arm']}"] = r.pop("_diff_field")
            keep[f"ref_{case['arm']}"] = r.pop("_f_ref_final")
            summary["stepwise"].append(r)
            print(f"{case['arm']} {case['model']:11s} tau={case['tau']:<4} "
                  f"eps={r['eps_scaled_inf']:.3e} pass={r['pass']} "
                  f"mass={r['mass_drift_rel']:.1e} "
                  f"mom={r['momentum_drift_abs_over_mass']:.1e}")
    rows, impl_fields = run_streaming_audit()
    summary["streaming_audit"] = rows
    for r in rows:
        print(f"X {r['impl']:28s} eps={r['eps_scaled_inf']:.3e} pass={r['pass']}")
    for case in dec["cases"]:
        if case["arm"].startswith("P"):
            r = run_physics(case)
            summary["physics"].append(r)
            print(f"{case['arm']} tau={case['tau']} err_max_vs_analytic="
                  f"{r['err_vs_analytic_max_rel']:.4%} rate_err={r['rate_err_rel']:.4%} "
                  f"pass={r['pass']}")
    np.savez_compressed(RUNS / "raw.npz", **keep,
                        **{f"impl_{k}": v for k, v in impl_fields.items()})
    (OUT / "summary.json").write_text(S.dumps(summary))

    lines = ["arm,kind,model,tau,steps,eps_scaled_inf,gate,pass,extra_metric,extra_value"]
    for r in summary["stepwise"]:
        lines.append(f"{r['arm']},stepwise,{r['model']},{r['tau']},{r['steps']},"
                     f"{r['eps_scaled_inf']:.6e},{r['gate']:.0e},{r['pass']},"
                     f"mass_drift_rel,{r['mass_drift_rel']:.3e}")
    for r in summary["streaming_audit"]:
        lines.append(f"{r['impl']},streaming,{r['lattice']},-,-,"
                     f"{r['eps_scaled_inf']:.6e},{r['gate']:.0e},{r['pass']},layout,{r['layout']}")
    for r in summary["physics"]:
        lines.append(f"{r['arm']},physics,srt,{r['tau']},{r['steps']},"
                     f"{r['err_vs_analytic_max_rel']:.6e},{r['gate']:.0e},{r['pass']},"
                     f"rate_err_rel,{r['rate_err_rel']:.3e}")
    (OUT / "table.csv").write_text("\n".join(lines) + "\n")

    n_pass = sum(1 for r in summary["stepwise"] if r["pass"])
    x_pass = sum(1 for r in rows if r["pass"])
    p_pass = sum(1 for r in summary["physics"] if r["pass"])
    print(f"\nTOTAL stepwise {n_pass}/{len(summary['stepwise'])}, "
          f"streaming {x_pass}/{len(rows)}, physics {p_pass}/{len(summary['physics'])}")
    return 0 if (n_pass == len(summary["stepwise"]) and x_pass == len(rows)
                 and p_pass == len(summary["physics"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())