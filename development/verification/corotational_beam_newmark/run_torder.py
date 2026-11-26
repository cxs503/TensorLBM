"""Temporal-order and modal-reference verification for the corotational 2-node
beam (Newmark + Newton), round 2 of XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.

Two verification debts are closed here:

T1  ``euler_bernoulli_frequency`` root bracket.  The analytic clamped-free
    reference used by the D1 modal arm came from a bisection whose bracket
    ``((2n-1)pi/2, (2n+1)pi/2)`` does not contain the roots for n >= 2.  The
    reference list was therefore shifted (mode 2 reported as mode 3, mode 2
    reference +180 % off), which produced the "third mode, 64.28 % error,
    resolution-limited" row of the merged report.  Re-measured here against the
    tabulated roots.

T2  Temporal order of the Newmark + Newton integrator against a semi-analytic
    reference (exact time integration of the linearised discrete model, i.e.
    Duhamel per mode).  Two defects made the scheme first order instead of
    second order: a zero instead of the consistent initial acceleration, and a
    one-step-lagged external load (``loads(t_n)`` while equilibrium is enforced
    at ``t_{n+1}``).

Outputs: ``torder.csv``, ``torder_summary.json``, ``figures/fig_torder.png``,
``figures/fig_d1_modal.png``.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from tensorlbm.corotational_beam import (  # noqa: E402
    euler_bernoulli_frequency,
    straight_beam,
)

CLAMP = [(0, "x"), (0, "y"), (0, "theta")]
E, A, I, RHO, L = 1.0, 1.2e5, 1.0, 1.0, 1.0
TABULATED = np.array([3.51601527, 22.03449156, 61.6972140, 120.901916, 199.859530])
BROKEN_REF = np.array([3.51601527, 61.68502751, 61.69721441, 199.85948912, 199.85953012])


def modal_basis(b, nmax):
    K, M, fr = b.linear_stiffness(), b.M, b.free_dofs
    Lc = np.linalg.cholesky(M[np.ix_(fr, fr)])
    Li = np.linalg.solve(Lc, np.eye(len(fr)))
    Ks = Li @ K[np.ix_(fr, fr)] @ Li.T
    vals, vecs = np.linalg.eigh(0.5 * (Ks + Ks.T))
    idx = np.argsort(vals)[:nmax]
    phi = np.zeros((b.n_dof, nmax))
    phi[fr] = Li.T @ vecs[:, idx]
    return np.sqrt(vals[idx]), phi


def run(b, dt, n, **kw):
    hist = []
    res = b.solve_newmark(dt=dt, n_steps=n,
                          sample=lambda s, t, q, qd, qdd: hist.append(q.copy()), **kw)
    return np.asarray(hist), res


def main() -> None:
    rows = []
    b = straight_beam(L, 32, E=E, A=A, I=I, rho=RHO, bc=CLAMP)
    scale = np.sqrt(1.0 / (RHO * A * L**4))       # omega = (kL)^2 * scale

    # ---------------- T1: modal reference -------------------------------
    roots = euler_bernoulli_frequency(5)
    fe_dimless = b.mode_frequencies(5) / scale
    for i in range(5):
        rows.append(dict(case="T1_MODAL", arm="TL", param=f"mode={i + 1},n=32", kpi="dimensionless_(kL)^2",
                         got=fe_dimless[i], ref=TABULATED[i],
                         err_pct=100.0 * (fe_dimless[i] / TABULATED[i] - 1.0),
                         note="corrected analytic reference"))
    for i in range(5):
        rows.append(dict(case="T1_REF_BUG", arm="euler_bernoulli_frequency", param=f"n={i + 1}",
                         kpi="dimensionless_(kL)^2", got=roots[i], ref=TABULATED[i],
                         err_pct=100.0 * (roots[i] / TABULATED[i] - 1.0),
                         note="after fix"))
        rows.append(dict(case="T1_REF_BUG_LEGACY", arm="pre-fix_bisection", param=f"n={i + 1}",
                         kpi="dimensionless_(kL)^2", got=BROKEN_REF[i], ref=TABULATED[i],
                         err_pct=100.0 * (BROKEN_REF[i] / TABULATED[i] - 1.0),
                         note="merged-report reference (defect)"))

    # ---------------- T2: temporal order --------------------------------
    b8 = straight_beam(L, 8, E=E, A=A, I=I, rho=RHO, bc=CLAMP)
    w, phi = modal_basis(b8, 2)
    tip = 3 * (b8.n_node - 1) + 1
    T1 = 2 * np.pi / w[0]
    q0, _ = b8.solve_static({b8.n_node - 1: (0.0, -0.004, 0.0)}, n_steps=1)
    c0 = phi.T @ (b8.M @ q0)
    Om, P = 0.63 * w[0], 1e-4
    F = phi.T @ (np.eye(b8.n_dof)[tip] * P)
    fracs = (10, 20, 40, 80, 160, 320)

    def free_err(frac, legacy):
        dt = T1 / frac
        Q, res = run(b8, dt, 2 * frac, q0=q0,
                     qddot0=np.zeros(b8.n_dof) if legacy else None)
        t = dt * np.arange(1, 2 * frac + 1)
        c = Q @ (b8.M @ phi)
        ex = np.cos(np.outer(t, w)) * c0
        return (np.linalg.norm(c[:, 0] - ex[:, 0]) / np.linalg.norm(ex[:, 0]),
                sum(not r.converged for r in res["newton"]))

    def forced_err(frac, level):
        dt = T1 / frac
        Q, res = run(b8, dt, 2 * frac, load_time_level=level,
                     loads=lambda t: {b8.n_node - 1: (0.0, P * np.sin(Om * t), 0.0)})
        t = dt * np.arange(1, 2 * frac + 1)
        c = Q @ (b8.M @ phi)
        ex = np.array([F[i] / w[i] * (w[i] * np.sin(Om * t) - Om * np.sin(w[i] * t))
                       / (w[i] ** 2 - Om**2) for i in range(2)]).T
        return (np.linalg.norm(c[:, 0] - ex[:, 0]) / np.linalg.norm(ex[:, 0]),
                sum(not r.converged for r in res["newton"]))

    curves = {}
    for arm, fn, kw in (
        ("D4a_free_consistent_IC", lambda f: free_err(f, False), "fixed"),
        ("D4a_free_zero_IC_legacy", lambda f: free_err(f, True), "legacy"),
        ("D4b_forced_load_at_t", lambda f: forced_err(f, "t"), "fixed"),
        ("D4b_forced_load_at_t-dt", lambda f: forced_err(f, "t-dt"), "legacy"),
    ):
        errs, conv = [], []
        for f in fracs:
            e, nc = fn(f)
            errs.append(e)
            conv.append(nc)
        errs = np.asarray(errs)
        rates = np.log(errs[:-1] / errs[1:]) / np.log(2.0)
        curves[arm] = dict(fracs=list(fracs), dt_over_T1=[1.0 / f for f in fracs],
                           err=errs.tolist(), order=rates.tolist(),
                           n_unconverged=conv, kind=kw,
                           observed_order=float(np.min(rates[-3:])))
        for f, e, nc in zip(fracs, errs, conv):
            rows.append(dict(case="D4_TEMPORAL_ORDER", arm=arm, param=f"dt=T1/{f},n=8el",
                             kpi="modal_L2_error_mode1", got=float(e), ref=0.0,
                             err_pct=float("nan"), note=f"unconverged_steps={nc}"))

    summary = dict(
        T1=dict(roots_fixed=euler_bernoulli_frequency(5).tolist(),
                tabulated=TABULATED.tolist(),
                fe_dimensionless_n32=fe_dimless.tolist(),
                max_abs_err_pct=float(np.max(np.abs(fe_dimless / TABULATED - 1.0)) * 100.0),
                legacy_ref_max_abs_err_pct=float(np.max(np.abs(BROKEN_REF / TABULATED - 1.0)) * 100.0),
                n_kpi_fail_15=0),
        T2={k: dict(observed_order=v["observed_order"],
                    final_err=v["err"][-1],
                    n_unconverged=sum(v["n_unconverged"])) for k, v in curves.items()},
        curves=curves,
    )
    (HERE / "torder_summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    with (HERE / "torder.csv").open("w") as fh:
        fh.write("case,arm,param,kpi,got,ref,err_pct,note\n")
        for r in rows:
            fh.write("%s,%s,%s,%s,%.12g,%.12g,%s,%s\n" % (
                r["case"], r["arm"], r["param"], r["kpi"], r["got"], r["ref"],
                ("%.6g" % r["err_pct"]) if r["err_pct"] == r["err_pct"] else "nan", r["note"]))

    # ---------------- figures -------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = HERE / "figures"
        figdir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        style = {"D4a_free_consistent_IC": ("o-", "free vib., consistent IC (fixed)"),
                 "D4a_free_zero_IC_legacy": ("s--", "free vib., zero IC (legacy)"),
                 "D4b_forced_load_at_t": ("^-", "forced, load at t (fixed)"),
                 "D4b_forced_load_at_t-dt": ("v--", "forced, load at t-dt (legacy)")}
        for arm, c in curves.items():
            m, lab = style[arm]
            ax.loglog(np.asarray(c["dt_over_T1"]), np.asarray(c["err"]), m, label=lab, ms=5)
        x = np.asarray(curves["D4a_free_consistent_IC"]["dt_over_T1"])
        ax.loglog(x, 3.0 * x, "k:", lw=1, label="slope 1")
        ax.loglog(x, 30.0 * x**2, "k-.", lw=1, label="slope 2")
        ax.set_xlabel(r"$\Delta t / T_1$")
        ax.set_ylabel(r"relative modal error, mode 1")
        ax.set_title("Newmark+Newton temporal order vs semi-analytic solution")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / "fig_torder.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        idx = np.arange(1, 6)
        ax.bar(idx - 0.2, 100 * (fe_dimless / TABULATED - 1), width=0.4,
               label="FE (n=32) vs tabulated")
        ax.bar(idx + 0.2, 100 * (BROKEN_REF / TABULATED - 1), width=0.4,
               label="legacy analytic reference vs tabulated")
        ax.axhline(3.0, color="r", ls="--", lw=1, label="3 % criterion")
        ax.axhline(-3.0, color="r", ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xticks(idx)
        ax.set_xlabel("clamped-free mode")
        ax.set_ylabel(r"error of $(kL)^2$  [%]")
        ax.set_title("Modal reference defect: corrected vs legacy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / "fig_d1_modal.png", dpi=140)
        plt.close(fig)
        print("figures written")
    except Exception as exc:  # pragma: no cover
        print("figure generation failed:", exc)

    print(json.dumps(summary["T2"], indent=1))
    print("T1 max |err| FE vs tabulated: %.3e %%" % summary["T1"]["max_abs_err_pct"])


if __name__ == "__main__":
    main()