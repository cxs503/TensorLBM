"""Round-3 verification of the D3 (nonlinear energy) arm of
XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.

What this closes
----------------
The merged round-2 report declared the D3 arm "not re-measured" and quoted a
single energy drift (0.0045 %).  Re-running the predeclared D3 configuration
showed that the drift numbers cannot be read the way the merged report reads
them, for three independent reasons that are measured here:

F7  **Silent censorship.** ``case_d3`` appended a KPI row only when the run
    reported ``converged``; a diverged predeclared point simply vanished from
    ``table.csv`` / ``summary.json`` (4 predeclared points, 3 rows, ``n_fail=0``).
    Fixed in ``run_xval.py`` (``record_fail`` + ``predeclared_points`` guard).

F8  **Step-size adequacy was never stated.**  The dynamic ladder is quoted as
    ``dt = T1/200``, but the discrete model's stiffest mode is *not* the
    fundamental: with ``A/I = 1.2e5`` the axial/rotational modes are three
    orders of magnitude faster than the fundamental, so ``T1/200`` is ~19 x the
    shortest discrete period at ``n = 16`` (``omega_max * dt ~ 121``).  The
    number measured here is ``omega_max * dt`` per ladder point, from the
    generalised eigenproblem of the linearised tangent, so the reader can see
    which points are resolved and which are only formally stable.

F9  **The D3 "energy" is not an invariant of the LIFE arm.**  The metric is
    ``KE + strain_energy`` of the *consistent* family; the LIFE force vector is
    not the gradient of that energy, so its drift measures the formulation
    mismatch, not energy conservation.  Both arms are reported separately and
    the parity of their tip trajectories is measured directly instead.

Measured here (all numbers are written to ``d3order.csv`` /
``d3order_summary.json``, no hand copy):

* per-arm, per-dt convergence status, ``max|q|``, energy drift;
* observed order of the energy drift in ``dt`` (non-dissipative 2nd-order
  scheme => drift ~ dt^2 up to the resolved-stiff-mode limit);
* observed order of the tip displacement error at ``t = T1`` against the
  finest ladder point (temporal order of the *nonlinear* transient, which was
  only ever measured for the linearised arms in round 2);
* TL-vs-LIFE trajectory parity in the nonlinear regime.

Run:  python development/verification/corotational_beam_newmark/run_d3order.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.corotational_beam import straight_beam  # noqa: E402

L = 1.0
EI = 1.0
A_SECTION = 1.2e5
RHOA = 1.0
ARMS = {"TL": "consistent", "LIFE": "life"}
CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
N_ELEM = 16
LAM = 5.0
FRACS = (200, 400, 800, 1600, 3200)
DIVERGED_FACTOR = 1e3          # predeclared stability rule of the case set
# Resolved-time sub-study (D3R): a fixed physical window short compared with the
# fundamental period (T1 = 619 s), stepped so that omega_max*dt <= 1.6.  The
# window makes the study affordable while the drift per period scales linearly
# with the window, so the *order* in dt and the arm-vs-arm parity are unaffected.
RESOLVED_WINDOW_S = 20.0
RESOLVED_DTS = (0.04, 0.02, 0.01, 0.005)


def beam(n_elem: int, arm: str):
    return straight_beam(L, n_elem, E=1.0, A=A_SECTION, I=EI, rho=RHOA,
                         bc=CLAMP, formulation=ARMS[arm])


def omega_max(b) -> float:
    """Stiffest frequency of the *linearised discrete* model.

    Generalised eigenproblem ``K_T(0) phi = w^2 M phi`` on the free DOFs,
    symmetrised by the diagonal of the consistent mass matrix (the mass matrix
    is diagonal-dominant here; ratio ``max diag / min diag ~ 2e4``, so the
    scaling is what makes the eigenvalues well conditioned in double precision).
    """
    fr = b.free_dofs
    K = b.tangent_stiffness(np.zeros(b.n_dof))[np.ix_(fr, fr)]
    M = b.M[np.ix_(fr, fr)]
    d = np.sqrt(np.diag(M))
    Ks = K / np.outer(d, d)
    ev = np.linalg.eigvalsh(0.5 * (Ks + Ks.T))
    return float(np.sqrt(np.clip(ev, 0.0, None)).max())


def main() -> int:
    rows: list[dict] = []
    summary: dict = {"config": dict(L=L, EI=EI, A=A_SECTION, rhoA=RHOA,
                                    n_elem=N_ELEM, lam=LAM, fracs=list(FRACS)),
                     "arms": {}, "step_adequacy": {}, "order": {}}
    t0 = time.time()

    for arm in ARMS:
        b = beam(N_ELEM, arm)
        w1 = b.mode_frequencies(1)[0]
        T1 = 2.0 * np.pi / w1
        wmax = omega_max(b)
        summary["step_adequacy"][arm] = dict(
            omega1=float(w1), T1=float(T1), omega_max=float(wmax),
            T_min=float(2.0 * np.pi / wmax), ratio_omega_max_over_omega1=float(wmax / w1),
            note="omega_max*dt > 1 means the stiffest discrete mode is not resolved "
                 "by the step; Newmark (beta=1/4, gamma=1/2) is non-dissipative but "
                 "the *Newton predictor* at such a step is far outside the "
                 "quadratic-convergence basin",
        )
        q0, _ = b.solve_static({b.n_node - 1: (0.0, -LAM * EI / L ** 2, 0.0)},
                               n_steps=20)
        summary["arms"].setdefault(arm, {})
        summary["arms"][arm]["q0_tip"] = b.tip_deflection(q0).tolist()
        summary["arms"][arm]["strain_energy_q0"] = float(b.strain_energy(q0))
        ladder = {}
        tip_at_T1 = {}
        for frac in FRACS:
            dt = T1 / frac
            e_hist = []

            def sample(step, t, q, qdot, qddot, e_hist=e_hist, b=b):
                e_hist.append(b.kinetic_energy(qdot) + b.strain_energy(q))

            if not np.isfinite(dt * wmax):
                raise RuntimeError("non-finite dt")
            res = b.solve_newmark(dt=dt, n_steps=int(frac), q0=q0, sample=sample)
            e = np.asarray(e_hist)
            max_q = float(np.abs(res["q_hist"]).max())
            diverged = (not np.isfinite(max_q)) or max_q > DIVERGED_FACTOR * L
            converged = bool(res["converged"]) and not diverged
            tip = float(b.tip_deflection(res["q"])[1] / L)
            d = dict(
                arm=arm, frac=frac, dt=float(dt), steps=int(frac),
                omega_max_dt=float(wmax * dt),
                converged=converged,
                max_abs_q_over_L=max_q,
                newton_max_iters=int(res["max_iterations"]),
                n_unconverged_steps=int(sum(1 for r in res["newton"] if not r.converged)),
                tip_y_over_L_at_T1=tip if converged else float("nan"),
                energy_drift_pct=(float(100.0 * abs(e[-1] - e[0]) / e[0])
                                  if converged else float("nan")),
            )
            ladder[frac] = d
            if converged:
                tip_at_T1[frac] = tip
            rows.append(dict(
                case="D3_ORDER", arm=arm, param=f"lambda={LAM},dt=T1/{frac},n={N_ELEM}",
                kpi="omega_max_dt", value=float(wmax * dt),
                status="PASS" if converged else "FAILED",
                note="stiffest-mode resolution diagnostic",
            ))
            rows.append(dict(
                case="D3_ORDER", arm=arm, param=f"lambda={LAM},dt=T1/{frac},n={N_ELEM}",
                kpi="energy_drift_pct", value=d["energy_drift_pct"],
                status="PASS" if converged else "FAILED",
                note="" if converged else
                     f"diverged/non-converged: max|q|={max_q:.2e}L, "
                     f"{d['n_unconverged_steps']}/{d['steps']} steps at maxit",
            ))
        summary["arms"][arm]["ladder"] = {str(k): v for k, v in ladder.items()}

        # --- observed order of the energy drift in dt -------------------
        ok = [ladder[f] for f in FRACS if ladder[f]["converged"]]
        if len(ok) >= 3:
            a, c = ok[-3], ok[-1]
            p = (np.log(a["energy_drift_pct"] / c["energy_drift_pct"])
                 / np.log(c["dt"] / a["dt"]))
            summary["order"][f"{arm}_energy_drift"] = dict(
                observed_order_p=float(p),
                note="drift ~ dt^p; a non-dissipative 2nd-order scheme gives p=2 "
                     "(p<2 means the drift is dominated by the unresolved stiff modes)",
            )
        # --- observed order of the nonlinear transient itself -----------
        fine = max(tip_at_T1) if tip_at_T1 else None
        if fine is not None:
            ref = tip_at_T1[fine]
            errs = {f: abs(tip_at_T1[f] - ref) for f in sorted(tip_at_T1)[:-1]
                    if f in tip_at_T1}
            summary["order"][f"{arm}_trajectory"] = dict(
                reference_dt=T1 / fine, reference_tip_y_over_L=ref,
                err_vs_reference={str(k): float(v) for k, v in errs.items()},
            )
            ks = sorted(errs)
            if len(ks) >= 2 and errs[ks[-1]] > 0:
                summary["order"][f"{arm}_trajectory"]["observed_order_p"] = float(
                    np.log(errs[ks[0]] / errs[ks[-1]])
                    / np.log((T1 / ks[0]) / (T1 / ks[-1]))
                ) if errs[ks[0]] > 0 else float("nan")

    # --- D3R: resolved-step energy conservation in the same nonlinear regime
    summary["resolved_window"] = dict(window_s=RESOLVED_WINDOW_S,
                                      dts=list(RESOLVED_DTS), arms={})
    for arm in ARMS:
        b = beam(N_ELEM, arm)
        w1 = b.mode_frequencies(1)[0]
        q0, _ = b.solve_static({b.n_node - 1: (0.0, -LAM * EI / L ** 2, 0.0)},
                               n_steps=20)
        block = {}
        for dt in RESOLVED_DTS:
            n_steps = int(round(RESOLVED_WINDOW_S / dt))
            e_hist = []

            def sample(step, t, q, qdot, qddot, e_hist=e_hist, b=b):
                e_hist.append(b.kinetic_energy(qdot) + b.strain_energy(q))

            res = b.solve_newmark(dt=dt, n_steps=n_steps, q0=q0, sample=sample)
            e = np.asarray(e_hist)
            max_q = float(np.abs(res["q_hist"]).max())
            converged = bool(res["converged"]) and np.isfinite(max_q) and max_q < DIVERGED_FACTOR
            d = dict(dt=float(dt), steps=n_steps, omega_max_dt=float(omega_max(b) * dt),
                     converged=converged, max_abs_q_over_L=max_q,
                     n_unconverged_steps=int(sum(1 for r in res["newton"] if not r.converged)),
                     tip_y_over_L_at_window=(float(b.tip_deflection(res["q"])[1] / L)
                                             if converged else float("nan")),
                     energy_drift_pct=(float(100.0 * abs(e[-1] - e[0]) / e[0])
                                       if converged else float("nan")))
            block[str(dt)] = d
            rows.append(dict(
                case="D3R_RESOLVED", arm=arm,
                param=f"lambda={LAM},dt={dt:g}s,window={RESOLVED_WINDOW_S:g}s,n={N_ELEM}",
                kpi="energy_drift_pct", value=d["energy_drift_pct"],
                status="PASS" if converged else "FAILED",
                note=f"omega_max*dt={d['omega_max_dt']:.2f}"
                     + ("" if converged else f"; {d['n_unconverged_steps']} steps at maxit"),
            ))
        summary["resolved_window"]["arms"][arm] = block
        ok = [block[str(dt)] for dt in RESOLVED_DTS if block[str(dt)]["converged"]]
        if len(ok) >= 3:
            a, c = ok[-3], ok[-1]
            summary.setdefault("order", {})[f"{arm}_D3R_energy_drift"] = dict(
                observed_order_p=float(np.log(a["energy_drift_pct"] / c["energy_drift_pct"])
                                       / np.log(c["dt"] / a["dt"])),
                note="resolved-step energy drift in the large-amplitude regime",
            )

    # --- TL vs LIFE parity in the nonlinear regime ----------------------
    parity = {}
    for frac in FRACS:
        tl = summary["arms"]["TL"]["ladder"].get(str(frac), {})
        li = summary["arms"]["LIFE"]["ladder"].get(str(frac), {})
        if tl.get("converged") and li.get("converged"):
            d = 100.0 * abs(tl["tip_y_over_L_at_T1"] - li["tip_y_over_L_at_T1"]) / abs(
                tl["tip_y_over_L_at_T1"])
            parity[str(frac)] = dict(
                tip_TL=tl["tip_y_over_L_at_T1"], tip_LIFE=li["tip_y_over_L_at_T1"],
                rel_diff_pct=float(d), within_owner_15pct=bool(d < 15.0),
            )
    summary["arm_parity_nonlinear"] = parity

    summary["wall_clock_s"] = time.time() - t0
    (HERE / "d3order_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    fields = ["case", "arm", "param", "kpi", "value", "status", "note"]
    with (HERE / "d3order.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    # ------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm, style in (("TL", "o-"), ("LIFE", "s--")):
            lad = summary["arms"][arm]["ladder"]
            xs = [lad[str(f)]["dt"] for f in FRACS if lad[str(f)]["converged"]]
            ys = [lad[str(f)]["energy_drift_pct"] for f in FRACS if lad[str(f)]["converged"]]
            ax[0].loglog(xs, ys, style, label=arm)
            bad = [lad[str(f)]["dt"] for f in FRACS if not lad[str(f)]["converged"]]
            if bad:
                ax[0].loglog(bad, [max(ys) if ys else 1.0] * len(bad), "rx",
                             label=f"{arm} diverged")
        ax[0].set_xlabel(r"$\Delta t$ [s]")
        ax[0].set_ylabel("energy drift over 1 period [%]")
        ax[0].set_title("D3 nonlinear energy drift vs dt (lambda=5, n=16)")
        ax[0].grid(True, which="both", alpha=0.3)
        ax[0].legend(fontsize=8)

        for arm in ARMS:
            lad = summary["arms"][arm]["ladder"]
            f = [str(x) for x in FRACS]
            ax[1].semilogx([lad[k]["dt"] for k in f],
                           [lad[k]["omega_max_dt"] for k in f], "o-", label=f"{arm}")
        ax[1].axhline(1.0, color="k", lw=1, ls=":")
        ax[1].set_xlabel(r"$\Delta t$ [s]")
        ax[1].set_ylabel(r"$\omega_{max}\,\Delta t$")
        ax[1].set_title("Stiffest-mode resolution of the D3 ladder")
        ax[1].grid(True, which="both", alpha=0.3)
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        (HERE / "figures").mkdir(exist_ok=True)
        fig.savefig(HERE / "figures" / "fig_d3_order.png", dpi=130)
        print("figure:", HERE / "figures" / "fig_d3_order.png")
    except Exception as exc:  # pragma: no cover
        print("figure skipped:", exc)

    print(json.dumps({k: summary[k] for k in ("order", "arm_parity_nonlinear")}, indent=1))
    print(f"wall clock {summary['wall_clock_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())