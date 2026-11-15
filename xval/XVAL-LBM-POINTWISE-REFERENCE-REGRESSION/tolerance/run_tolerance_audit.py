#!/usr/bin/env python3
"""Round 4 audit: dtype noise floor and detection limit of the pointwise gate.

Produces, next to this file:
    summary.json          every number quoted in the report
    table.csv             one row per (tau, precision mode, step-30 result)
    sensitivity.csv       eps_inf vs relative tau perturbation, per arithmetic
    tolerance_policy.json the admissible threshold and the detection floor per
                          precision mode -- the machine-readable policy the
                          gate (and the test that guards it) should read
    fig_eps_vs_step.png   deviation-from-reference growth, per precision mode
    fig_sensitivity.png   eps_inf vs injected tau regression, with noise bands

Run:  python xval/XVAL-LBM-POINTWISE-REFERENCE-REGRESSION/tolerance/run_tolerance_audit.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import precision_emulation as pe  # noqa: E402

BUNDLE = pe.D3 / "refdata" / "pointwise_d3q19_v1.npz"
MANIFEST = json.loads((pe.D3 / "refdata" / "pointwise_d3q19_v1.manifest.json").read_text())
TAUS = (0.55, 0.7, 1.0)
STEPS = MANIFEST["config"]["steps"]
#: precision modes; ``(arithmetic, storage)``
MODES = [
    ("fp64", None),
    ("fp64", "fp32"),
    ("fp64", "bf16"),
    ("fp64", "fp16"),
    ("fp32", None),
    ("fp32", "fp32"),
]
GATE_FP64 = 1e-10  #: threshold the frozen gate uses today (test_xval_pointwise_d3q19_refdata)
DELTAS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2)


def mode_name(arith: str, storage) -> str:
    return arith if storage is None else f"{arith}+{storage}_store"


def extra_fields() -> dict[str, np.ndarray]:
    """Second initial field from the round-3 generator (independent of seed 0)."""
    if str(pe.XV) not in sys.path:
        sys.path.insert(0, str(pe.XV))
    try:
        import lbmpy_pointwise_ref as gen  # noqa: PLC0415
    except Exception:  # pragma: no cover - lbmpy absent on some nodes
        return {}
    f = gen.make_test_field("D3Q19", tau=0.7, seed=1, amplitude=0.05)
    return {"seed1_tau0.7": np.asarray(f, dtype=np.float64)}


def gate_from_floor(floor: float, margin: float = 100.0) -> float:
    """Smallest round threshold >= ``margin``x the floor (1 significant digit)."""
    if floor <= 0.0:
        return GATE_FP64
    target = margin * floor
    exp = int(np.floor(np.log10(target)))
    return float(np.ceil(target / 10.0**exp) * 10.0**exp)


def main() -> int:
    refs0 = pe.load_bundle(BUNDLE)
    fields = {"seed0": refs0["f0"]}
    fields.update(extra_fields())
    print(f"fields: {list(fields)}", flush=True)

    # fp64 reference trajectory per (field, tau) -- the target of every deviation
    ref = {}
    for fname, f0 in fields.items():
        for tau in TAUS:
            ref[(fname, tau)] = pe.run(f0, tau, STEPS, "fp64", None)
    # control: the fp64 reference operator must reproduce the frozen bundle
    ctrl = {
        "eps_inf_t1": pe.core.compare(ref[("seed0", 0.7)][0], refs0["ref_t1"])["eps_inf"],
        "eps_inf_t30": pe.core.compare(ref[("seed0", 0.7)][-1], refs0["ref_t30"])["eps_inf"],
    }
    print(f"control fp64 vs frozen bundle: {ctrl}", flush=True)

    rows, curves = [], {}
    for fname, f0 in fields.items():
        for tau in TAUS:
            for arith, storage in MODES:
                name = mode_name(arith, storage)
                traj = pe.run(f0, tau, STEPS, arith, storage)
                target = ref[(fname, tau)]
                # endpoints pinned to the FROZEN bundle where it exists, so the
                # fp64 mode is measured against lbmpy's reference, not against
                # its own code (which would be 0 by construction)
                targets = {"_traj": target}
                if fname == "seed0" and tau == MANIFEST["config"]["tau"]:
                    targets["ref_t1"] = refs0["ref_t1"]
                    targets["ref_t30"] = refs0["ref_t30"]
                eps = [r["eps_inf"] for r in pe.eps_curve(traj, targets)]
                curves.setdefault(name, []).append(
                    {"field": fname, "tau": tau, "eps": eps})
                rows.append({
                    "field": fname, "tau": tau, "mode": name,
                    "arithmetic": arith, "storage": str(storage),
                    "eps_inf_t1": eps[0], "eps_inf_t30": eps[-1],
                    "eps_inf_max": max(eps),
                    "mass_drift_t30": abs(float(traj[-1].sum() - f0.sum())) / abs(float(f0.sum())),
                })
    # production code path (torch fp64 repo arm) under storage rounding only
    repo_rows = []
    for tau in (0.7,):
        for storage in (None, "fp32", "fp16"):
            traj = pe.run_repo_arm("A1_solver3d_bgk", fields["seed0"], tau, STEPS, storage)
            ep = [pe.core.compare(a, b)["eps_inf"]
                  for a, b in zip(traj, ref[("seed0", tau)])]
            repo_rows.append({"arm": "A1_solver3d_bgk", "tau": tau,
                              "storage": str(storage),
                              "eps_inf_t1": ep[0], "eps_inf_t30": ep[-1],
                              "eps_inf_max": max(ep)})
            print(f"repo A1 storage={storage}: eps_max={max(ep):.3e}", flush=True)

    # --- noise floor per mode (all fields, all taus) -----------------------
    policy = {}
    for arith, storage in MODES:
        name = mode_name(arith, storage)
        sel = [r for r in rows if r["mode"] == name]
        floor = max(r["eps_inf_max"] for r in sel)
        policy[name] = {
            "arithmetic": arith,
            "storage": None if storage is None else str(storage),
            "noise_floor_eps_inf": floor,
            "noise_floor_where": max(sel, key=lambda r: r["eps_inf_max"])
            ["field"] + f"@tau={max(sel, key=lambda r: r['eps_inf_max'])['tau']}",
            "admissible_gate_eps_inf": gate_from_floor(floor),
        }
    policy["fp64"]["admissible_gate_eps_inf"] = GATE_FP64  # keep the shipped gate
    policy["fp64"]["gate_margin_over_floor"] = (
        GATE_FP64 / policy["fp64"]["noise_floor_eps_inf"]
        if policy["fp64"]["noise_floor_eps_inf"] > 0 else None)
    policy["repo_A1_solver3d_bgk+fp32_store"] = {
        "arithmetic": "torch float64", "storage": "fp32",
        "noise_floor_eps_inf": max(r["eps_inf_max"] for r in repo_rows
                                   if r["storage"] == "fp32"),
        "admissible_gate_eps_inf": gate_from_floor(
            max(r["eps_inf_max"] for r in repo_rows if r["storage"] == "fp32")),
    }

    # --- sensitivity: how small a tau drift can each mode still see? -------
    sens = {}
    for arith, storage in (("fp64", None), ("fp64", "fp32"), ("fp32", None)):
        name = mode_name(arith, storage)
        rows_d = []
        for d in DELTAS:
            tr = pe.run(fields["seed0"], 0.7 * (1.0 + d), STEPS, arith, storage)
            e = pe.core.compare(tr[-1], ref[("seed0", 0.7)][-1])["eps_inf"]
            rows_d.append({"delta_rel": d, "eps_inf": e})
            print(f"sensitivity {name} delta={d:.0e} eps={e:.3e}", flush=True)
        gate = policy[name]["admissible_gate_eps_inf"]
        gate10 = gate_from_floor(policy[name]["noise_floor_eps_inf"], margin=10.0)
        snr1pct = next(r["eps_inf"] for r in rows_d if r["delta_rel"] == 1e-2) / max(
            policy[name]["noise_floor_eps_inf"], 1e-300)
        sens[name] = {"rows": rows_d,
                      "detect": pe.detect_floor(rows_d, gate),
                      "detect_margin10": pe.detect_floor(rows_d, gate10),
                      "gate_eps_inf": gate, "gate_eps_inf_margin10": gate10,
                      "snr_at_1pct_tau": snr1pct}
        policy[name]["detectable_tau_rel"] = sens[name]["detect"]["delta_rel"]
        policy[name]["admissible_gate_eps_inf_margin10"] = gate10
        policy[name]["detectable_tau_rel_margin10"] = sens[name]["detect_margin10"]["delta_rel"]
        policy[name]["snr_at_1pct_tau"] = snr1pct

    # --- does the floor grow with trajectory length? (scope of the policy) --
    extend = {}
    for arith, storage, name in (("fp64", None, "fp64"), ("fp32", None, "fp32"),
                                 ("fp64", "fp32", "fp64+fp32_store")):
        long_ref = pe.run(fields["seed0"], 0.7, 120, "fp64", None)
        long_run = pe.run(fields["seed0"], 0.7, 120, arith, storage)
        extend[name] = {str(n): pe.core.compare(long_run[n - 1], long_ref[n - 1])["eps_inf"]
                        for n in (30, 60, 120)}
        print(f"length dependence {name}: {extend[name]}", flush=True)

    summary = {
        "anchor": "XVAL-LBM-POINTWISE-REFERENCE-REGRESSION",
        "round": 4,
        "question": "dtype noise floor of the pointwise reference gate, and the "
                    "smallest tau regression it can still resolve per precision mode",
        "provenance": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "bundle": {"path": str(BUNDLE.relative_to(pe.REPO)),
                       "sha256": MANIFEST["sha256"],
                       "tau": MANIFEST["config"]["tau"], "steps": STEPS},
            "gate_under_audit": {"file": "tests/test_xval_pointwise_d3q19_refdata.py",
                                 "threshold_eps_inf": GATE_FP64},
            "supersedes": "xval/<anchor>/summary.json : fp32_sensitivity[*].eps_scaled_inf_fp32_rounding "
                          "== 0.0 is a tautology (both sides cast from fp64 arrays that already agree); "
                          "round 4 measures the reduced precision inside the time loop instead.",
        },
        "control_fp64_reproduces_frozen": ctrl,
        "taus": list(TAUS),
        "steps": STEPS,
        "fields": sorted(fields),
        "modes": [mode_name(a, s) for a, s in MODES],
        "table": rows,
        "repo_path": repo_rows,
        "policy": policy,
        "sensitivity": sens,
        "step_length_dependence": extend,
        "arith_dtype_check": {
            arith: pe.step_intermediate_dtypes(refs0["f0"], 0.7, arith)
            for arith in ("fp64", "fp32")
        },
        "curves": {k: [{kk: (vv if kk != "eps" else [float(x) for x in vv])
                        for kk, vv in c.items()} for c in v] for k, v in curves.items()},
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (HERE / "tolerance_policy.json").write_text(json.dumps(policy, indent=1) + "\n")

    with (HERE / "table.csv").open("w") as fh:
        cols = ["field", "tau", "mode", "arithmetic", "storage",
                "eps_inf_t1", "eps_inf_t30", "eps_inf_max", "mass_drift_t30"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    with (HERE / "sensitivity.csv").open("w") as fh:
        fh.write("mode,delta_rel,eps_inf,gate_eps_inf,detected\n")
        for name, s in sens.items():
            for r in s["rows"]:
                fh.write(f"{name},{r['delta_rel']},{r['eps_inf']},{s['gate_eps_inf']},"
                         f"{r['eps_inf'] > s['gate_eps_inf']}\n")

    _figures(curves, sens, policy)
    print("\nnoise floors:")
    for k, v in policy.items():
        print(f"  {k:32s} floor={v['noise_floor_eps_inf']:.3e}"
              + (f" gate={v['admissible_gate_eps_inf']:.1e}"
                 f" detectable_tau_rel={v.get('detectable_tau_rel')}"
                 if 'admissible_gate_eps_inf' in v else ""))
    return 0


def _figures(curves, sens, policy) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"fp64": "k", "fp64+fp32_store": "C0", "fp64+bf16_store": "C1",
              "fp64+fp16_store": "C2", "fp32": "C3", "fp32+fp32_store": "C4"}
    for name, cs in curves.items():
        c = next((c for c in cs if c["field"] == "seed0" and c["tau"] == 0.7), cs[0])
        ax.semilogy(range(1, len(c["eps"]) + 1), np.maximum(c["eps"], 1e-18),
                    label=name, color=colors.get(name), lw=1.5)
    ax.axhline(policy["fp64"]["admissible_gate_eps_inf"], color="k", ls="--", lw=1,
               label=f"shipped fp64 gate {policy['fp64']['admissible_gate_eps_inf']:.0e}")
    fp32_gate = policy["fp32"]["admissible_gate_eps_inf"]
    ax.axhline(fp32_gate, color="C3", ls="--", lw=1, label=f"fp32 gate proposal {fp32_gate:.0e}")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\varepsilon_\infty$ vs fp64 reference")
    ax.set_title("D3Q19 pointwise gate: deviation growth per precision mode (tau=0.7)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(HERE / "fig_eps_vs_step.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, s in sens.items():
        d = np.array([r["delta_rel"] for r in s["rows"]])
        e = np.array([max(r["eps_inf"], 1e-18) for r in s["rows"]])
        ax.loglog(d, e, "o-", label=f"{name} (gate {s['gate_eps_inf']:.0e})")
        ax.axhline(policy[name]["noise_floor_eps_inf"], color="grey", ls=":", lw=.8)
    ax.set_xlabel(r"injected relaxation-time regression $\delta\tau/\tau$")
    ax.set_ylabel(r"$\varepsilon_\infty$ at t=30")
    ax.set_title("Detection limit of the pointwise gate: signal vs dtype noise floor (grey dots)")
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(HERE / "fig_sensitivity.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())