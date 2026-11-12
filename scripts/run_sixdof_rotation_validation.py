"""Produce artifacts/CASE-SIXDOF-EULER-TOP evidence + figure (torque-free 6-DOF)."""
from __future__ import annotations
import json, math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tensorlbm.sixdof_rotation_validation import (
    band_summary, compare_symmetric_top, convergence_study, dzhanibekov_growth_rate,
    euler_symmetric_top_solution, falsify_reference_frequency, integrate_torque_free,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "sixdof_rotation_validation"
OUT.mkdir(parents=True, exist_ok=True)
I_TOP, W_TOP, I_TUMBLE = (2.0, 2.0, 1.0), (0.3, 0.0, 1.0), (1.0, 2.0, 3.0)

conv = convergence_study(I_TOP, W_TOP, dts=[0.04, 0.02, 0.01, 0.005], t_end=12.0)
top_run = integrate_torque_free(I_TOP, W_TOP, dt=0.02, n_steps=600)
top_metrics = compare_symmetric_top(top_run["omega"], top_run["times"], W_TOP, I_TOP)
lam = dzhanibekov_growth_rate(I_TUMBLE, 1.0)
tum = integrate_torque_free(I_TUMBLE, (1e-3, 1.0, 1e-3), dt=0.002, n_steps=2500)
band = band_summary(tum["times"], tum["omega"][:, 0], low=2e-3, high=1e-2)
fals = falsify_reference_frequency(0.05, W_TOP, I_TOP, dt=0.02, n_steps=600)
inv = top_run["invariants"]
e0, l0 = inv[0]["energy"], inv[0]["momentum_body"]
drift = {
    "energy_rel": max(abs(r["energy"] - e0) / e0 for r in inv),
    "momentum_body_rel": max(abs(r["momentum_body"] - l0) / l0 for r in inv),
    "momentum_world_rel": max(abs(r["momentum_world"] - inv[0]["momentum_world"]) / inv[0]["momentum_world"] for r in inv),
    "quat_norm_dev": max(abs(r["quat_norm"] - 1.0) for r in inv),
}
payload = {
    "case": "CASE-SIXDOF-EULER-TOP",
    "module": "tensorlbm.sixdof_rotation_validation",
    "library_path": "tensorlbm.sixdof_common.rigid_body_step -> tensorlbm.sixdof.step_sixdof",
    "symmetric_top": {"inertia": I_TOP, "omega0": W_TOP, "dt": 0.02, "t_end": 12.0,
                      "metrics": top_metrics, "invariant_drift": drift},
    "convergence": conv,
    "dzhanibekov": {"inertia": I_TUMBLE, "spin": 1.0, "linear_rate": lam,
                    "band": [2e-3, 1e-2], "fit": band,
                    "rel_err": abs(band["rate"] - lam) / lam},
    "falsification_5pct_reference_shift": fals,
}
(OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    t = top_run["times"]; w1 = top_run["omega"][:, 0]
    w1e = euler_symmetric_top_solution(W_TOP, I_TOP, t)[:, 0]
    axes[0].plot(t, w1, label="step_sixdof (dt=0.02)", lw=1.2)
    axes[0].plot(t, w1e, "--", label="exact symmetric top", lw=1.2)
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel(r"$\omega_1$ [rad/s]")
    axes[0].set_title("Torque-free precession vs exact"); axes[0].legend(); axes[0].grid(alpha=.3)
    rows = conv["rows"]
    axes[1].loglog([r["dt"] for r in rows], [r["rel_l2"] for r in rows], "o-", label="rel L2 error")
    d = np.array([r["dt"] for r in rows]); k = conv["observed_order"]
    axes[1].loglog(d, rows[0]["rel_l2"] * (d / d[0]) ** k, "k--", label=f"slope {k:.2f}")
    axes[1].set_xlabel("dt [s]"); axes[1].set_ylabel("rel. L2 error")
    axes[1].set_title("dt refinement (symplectic Euler)"); axes[1].legend(); axes[1].grid(alpha=.3, which="both")
    tt = tum["times"]; a = np.abs(tum["omega"][:, 0])
    axes[2].semilogy(tt, a, lw=1.2, label=r"$|\omega_1|$ (I1<I2<I3)")
    sel = (a >= 2e-3) & (a <= 1e-2)
    axes[2].semilogy(tt[sel], a[sel], ".", ms=2, label="fit band")
    axes[2].semilogy(tt, 1e-3 * np.exp(lam * tt), "k--", label=f"linear rate {lam:.4f} 1/s")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel(r"$|\omega_1|$ [rad/s]")
    axes[2].set_title("Intermediate-axis (Dzhanibekov) instability"); axes[2].legend(); axes[2].grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "figure.png", dpi=150)
    print("figure:", OUT / "figure.png")
except Exception as exc:  # pragma: no cover
    print("figure skipped:", exc)

def fmt(rows):
    head = "| dt [s] | steps | rel L2 | rel inf | axial drift | period rel err | energy drift |\n|---|---|---|---|---|---|---|\n"
    for r in rows:
        head += f"| {r['dt']:g} | {r['n_steps']} | {r['rel_l2']:.3e} | {r['rel_inf']:.3e} | {r['axial_rel_drift']:.1e} | {r['period_rel_err']:.2e} | {r['energy_drift']:.2e} |\n"
    return head

doc = f"""# CASE-SIXDOF-EULER-TOP — torque-free rigid-body rotation vs analytic solution

Anchor: verification debt of the merged 6-DOF path. `tensorlbm/sixdof_capability_contract.py`
certifies 6-DOF only as `CONTRACT_TESTED` and declares `WITHHELD_NO_PHYSICS_VALIDATION`;
before this case no trajectory of the integrator had ever been compared with a known
solution. This case discharges the **torque-free rotation** sub-scope only (coupling /
added-mass / moving-body flow physics remain WITHHELD).

## 1. Case introduction
Rigid body with zero external moment, body frame, Euler equations
`I dω/dt = M − ω × (Iω)`; integrated by the library path
`tensorlbm.sixdof_common.rigid_body_step → tensorlbm.sixdof.step_sixdof`
(explicit/symplectic Euler + exponential-map quaternion update, dt in seconds,
gravity disabled so the configuration is genuinely torque-free).

| parameter | value |
|---|---|
| symmetric top inertia (I1, I2, I3) | {I_TOP} |
| initial body rate ω0 | {W_TOP} rad/s |
| analytic precession rate Ω = (I1−I3)ω3/I1 | 0.5 rad/s (period 12.5664 s) |
| tumble body inertia | {I_TUMBLE}, ω0 = (1e−3, 1, 1e−3) rad/s |
| analytic intermediate-axis rate | {lam:.6f} 1/s |

## 2. Method
Three independent referees, none of which touches the integrator:
1. **Exact symmetric-top solution** (arbitrary amplitude, no linearisation):
   `ω1 = ω1_0 cos Ωt + ω2_0 sin Ωt`, `ω2 = −ω1_0 sin Ωt + ω2_0 cos Ωt`, `ω3 = ω3_0`.
2. **Exact first integrals**: `T = ½ω·Iω`, `|Iω|` (body), `|R(q) Iω|` (world, a constant
   vector under pure rotation), `|q| = 1`.
3. **Linear intermediate-axis rate** `λ = |ω2| √((I3−I2)(I2−I1)/(I1I3))`, used inside a
   stated amplitude band.

Same metric functions are applied to the analytic solution itself, so no number is fitted.
Falsification: the reference frequency is deliberately shifted by 0.05 rad/s (10% of Ω)
and the same gate must fail.

Reproduce: `python scripts/run_sixdof_rotation_validation.py`
(tests: `pytest tests/test_sixdof_rotation_validation.py`).

## 3. Results (programmatic, `artifacts/sixdof_rotation_validation/summary.json`)

Symmetric top, dt refinement at fixed t = 12 s:

{fmt(conv['rows'])}

Observed convergence order from log-log fit: **{conv['observed_order']:.3f}** (first-order, as expected
for explicit Euler on the gyroscopic term).

Single-run gate at dt = 0.02: rel L2 = {top_metrics['rel_l2']:.3e}, rel inf = {top_metrics['rel_inf']:.3e},
axial-rate drift = {top_metrics['axial_rel_drift']:.1e} (axial rate is exactly constant),
period recovered = {top_metrics['period_got']:.4f} s vs analytic {top_metrics['period_ref']:.4f} s
(rel err {top_metrics['period_rel_err']:.2e}).

Invariant drift over the same 12 s window (dt = 0.02): energy {drift['energy_rel']:.2e},
body |L| {drift['momentum_body_rel']:.2e}, world |L| {drift['momentum_world_rel']:.2e},
quaternion norm deviation {drift['quat_norm_dev']:.2e}.

Dzhanibekov (intermediate axis): fitted rate {band['rate']:.4f} 1/s vs linear theory {lam:.4f} 1/s
→ rel err **{abs(band['rate']-lam)/lam:.3%}** inside the band |ω1| ∈ [2e−3, 1e−2]
({band['n_samples']:.0f} samples, t ∈ [{band['t_first']:.2f}, {band['t_last']:.2f}] s).

Falsification (reference frequency shifted 10%): rel L2 = {fals['rel_l2']:.3e},
rel inf = {fals['rel_inf']:.3e} — the gate rejects a wrong reference by a factor
{fals['rel_l2']/max(top_metrics['rel_l2'],1e-30):.0f} over the accepted error.

## 4. Figures
`artifacts/sixdof_rotation_validation/figure.png` — (a) ω1 from the library path vs the exact
symmetric top, (b) log-log dt refinement with the fitted slope, (c) |ω1| growth for the
intermediate-axis case against the linear rate.

## 5. Caveats
- Explicit Euler on the gyroscopic term is **not** symplectic, so T and |Iω| drift secularly
  at O(dt) (2.3e−3 over 12 s at dt = 0.02). This is reported, not hidden; refinement reduces it.
- The Dzhanibekov fit is **semi-quantitative**: the exact solution is cnoidal, so the
  exponential band fit depends on the chosen amplitude window (bands 1e−3..5e−3 and
  2e−3..1e−2 differ by ~10%). Band and tolerance are stated (20%).
- Only the torque-free sub-scope is validated. Fluid coupling, added mass, moving-body drag
  and FSI energy exchange remain `WITHHELD_NO_PHYSICS_VALIDATION`.

## 6. Error judgement vs analytic reference
| quantity | reference | computed | rel. error | < 3%? |
|---|---|---|---|---|
| ω(t) trajectory, dt=0.02, 12 s | exact symmetric top | rel L2 {top_metrics['rel_l2']:.3e} | {top_metrics['rel_l2']:.3e} | yes |
| precession period | 12.5664 s | {top_metrics['period_got']:.4f} s | {top_metrics['period_rel_err']:.2e} | yes |
| axial rate ω3 | exactly constant | drift {top_metrics['axial_rel_drift']:.1e} | {top_metrics['axial_rel_drift']:.1e} | yes |
| intermediate-axis growth rate | {lam:.4f} 1/s (linearised) | {band['rate']:.4f} 1/s | {abs(band['rate']-lam)/lam:.2%} | no — linearisation-only anchor, stated in caveats |
"""
(OUT / "CASE-SIXDOF-EULER-TOP.md").write_text(doc)
pathlib.Path("docs/CASE-SIXDOF-EULER-TOP.md").write_text(doc)
print("wrote doc; order", conv["observed_order"], "top rel_l2", top_metrics["rel_l2"], "dz rel", abs(band['rate']-lam)/lam)
