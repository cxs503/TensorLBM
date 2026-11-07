#!/usr/bin/env python3
"""BMAN-2: ship zig-zag (overshoot) manoeuvre -- two-route verification + IMO gate.

Verifies the closed-loop helm capability added to ``tensorlbm.maneuvering``
(``simulate_zigzag`` / ``analytic_zigzag`` / ``check_imo_msc137_zigzag``) on the
KVLCC2-class linear sway-yaw model:

  1. **Two independent routes** agree on the overshoot angles and on the helm
     reversal / heading-peak times:
     * ``simulate_zigzag``  -- event-driven RK4 (switch and peak times located by
       bisection *inside* a step, so no O(dt) switch-time quantisation);
     * ``analytic_zigzag``  -- exact piecewise-modal solution (closed-form roots).
  2. **Order of accuracy**: time-step convergence of the overshoot angle against
     the analytic value (semi-implicit Euler -> first order; RK4 -> much faster).
  3. **Physical properties** of the linear model: rudder-angle scale invariance
     (20/20 = 2 x 10/10 in every angle and time) and port/starboard mirror
     symmetry.  A representative cross-flow (Y_vv/N_vv) model is run to show the
     invariance breaking, as it does on a real hull.
  4. **IMO MSC.137(76) overshoot gate**: implemented limits (10/10 -> 10 deg if
     L/V < 10 s else 20 deg; 20/20 -> 25 deg first / 40 deg second) applied to
     the nominal tanker (pass) and to a yaw-damping degradation family showing
     where the gate starts to fail.

Evidence boundary: the derivatives are a **representative** linear KVLCC2-class
set (not PMM/CFD-extracted), there is no rudder-rate limit, no propeller
slipstream, no heel/roll coupling.  The claim is *internal verification* of the
helmsman logic and of the criterion implementation -- absolute overshoot angles
for a specific ship remain WITHHELD.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from tensorlbm.maneuvering import (  # noqa: E402
    IMO_MSC137_SOURCE,
    NOMINAL_TANKER_PRIME,
    ManeuveringModel,
    analytic_zigzag,
    check_imo_msc137_zigzag,
    imo_msc137_zigzag_limits,
    prime_to_dimensional,
    simulate_zigzag,
)

# --------------------------------------------------------------------------- #
# ship (KVLCC2-class large tanker, full scale)
# --------------------------------------------------------------------------- #
L = 320.0
B = 58.0
D = 20.8
CB = 0.81
RHO = 1000.0
U = 7.97
MASS = RHO * CB * L * B * D
IZ = MASS * (0.25 * L) ** 2
L_OVER_V = L / U  # 40.2 s -> the 10/10 limit is the "L/V >= 10 s" branch (20 deg)

TESTS_DEG = (10.0, 20.0)
EULER_DTS = (1.0, 0.5, 0.25, 0.125)
RK4_DTS = (0.5, 0.25, 0.125, 0.0625)
REFERENCE_DT = 0.05

ROUTE_TOL_REL = 1e-4  # two-route agreement at the reference time step
RK4_TOL_REL = 1e-5
SCALE_TOL_REL = 1e-9


def build_model(prime: dict | None = None, **overrides) -> ManeuveringModel:
    p = dict(NOMINAL_TANKER_PRIME) if prime is None else dict(prime)
    dim = prime_to_dimensional(p, L, D, U, RHO)
    kwargs = dict(L=L, d=D, U=U, mass=MASS, inertia_z=IZ, rho=RHO, **dim)
    kwargs.update(overrides)
    return ManeuveringModel(**kwargs)


def rel_err(num: float, ref: float) -> float:
    return abs(num - ref) / abs(ref)


def nonlin_model(fac: float = 1.0) -> ManeuveringModel:
    """Same tanker plus a representative cross-flow pair scaled by ``fac``.

    ``Y_vv = fac*|Y_v|/U`` -- dimensionally consistent (force per (m/s)^2) and
    sized so the cross-flow force at ``v = U`` is ``fac`` times the linear
    ``Y_v v`` term.
    """
    dim = prime_to_dimensional(NOMINAL_TANKER_PRIME, L, D, U, RHO)
    return build_model(Y_vv=fac * abs(dim["Y_v"]) / U, N_vv=fac * abs(dim["N_v"]) / U)


def observed_order(dts, errs) -> float:
    """Least-squares slope of log(err) vs log(dt)."""
    x = np.log(np.asarray(dts, dtype=float))
    y = np.log(np.asarray(errs, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main() -> int:
    model = build_model()
    results: dict = {
        "benchmark": "maneuvering_zigzag",
        "ship": {
            "name": "KVLCC2-class tanker (representative linear derivatives)",
            "L_m": L,
            "B_m": B,
            "T_m": D,
            "Cb": CB,
            "U_mps": U,
            "L_over_V_s": L_OVER_V,
        },
        "imo_source": IMO_MSC137_SOURCE,
        "reference_dt": REFERENCE_DT,
        "routes": {},
        "convergence": {},
        "invariance": {},
        "crossflow": {},
        "imo": {},
        "verdict": {},
    }

    # ------------------------------------------------------------------ #
    # 1. two routes
    # ------------------------------------------------------------------ #
    for delta in TESTS_DEG:
        ana = analytic_zigzag(model, delta)
        sim = simulate_zigzag(model, delta, dt=REFERENCE_DT, integrator="rk4")
        key = f"{delta:.0f}/{delta:.0f}"
        rows = []
        for i in range(2):
            rows.append(
                {
                    "quantity": f"overshoot_{i + 1}_deg",
                    "sim": sim.overshoots_deg[i],
                    "analytic": ana.overshoots_deg[i],
                    "rel_err": rel_err(sim.overshoots_deg[i], ana.overshoots_deg[i]),
                }
            )
        for i in range(2):
            rows.append(
                {
                    "quantity": f"reversal_time_{i + 1}_s",
                    "sim": sim.reversal_times[i],
                    "analytic": ana.reversal_times[i],
                    "rel_err": rel_err(sim.reversal_times[i], ana.reversal_times[i]),
                }
            )
            rows.append(
                {
                    "quantity": f"peak_time_{i + 1}_s",
                    "sim": sim.peak_times[i],
                    "analytic": ana.peak_times[i],
                    "rel_err": rel_err(sim.peak_times[i], ana.peak_times[i]),
                }
            )
        results["routes"][key] = {
            "rows": rows,
            "max_rel_err": max(r["rel_err"] for r in rows),
            "analytic": ana.as_dict(),
            "sim": sim.as_dict(),
        }

    route_max = max(v["max_rel_err"] for v in results["routes"].values())

    # ------------------------------------------------------------------ #
    # 2. time-step convergence (Euler first order, RK4 fast)
    # ------------------------------------------------------------------ #
    for delta in TESTS_DEG:
        ana_ovs = analytic_zigzag(model, delta).first_overshoot_deg
        conv = {}
        for name, dts in (("semi_implicit_euler", EULER_DTS), ("rk4", RK4_DTS)):
            errs = []
            vals = []
            for dt in dts:
                r = simulate_zigzag(model, delta, dt=dt, integrator=name)
                vals.append(r.first_overshoot_deg)
                errs.append(abs(r.first_overshoot_deg - ana_ovs))
            conv[name] = {
                "dt": list(dts),
                "first_overshoot_deg": vals,
                "abs_err_deg": errs,
                "rel_err": [e / abs(ana_ovs) for e in errs],
                "observed_order": observed_order(dts, errs),
            }
        conv["analytic_first_overshoot_deg"] = ana_ovs
        results["convergence"][f"{delta:.0f}/{delta:.0f}"] = conv

    euler_orders = [c["semi_implicit_euler"]["observed_order"] for c in results["convergence"].values()]
    rk4_finest = [
        c["rk4"]["rel_err"][-1] for c in results["convergence"].values()
    ]
    order_ok = all(0.8 <= o <= 1.3 for o in euler_orders)
    rk4_ok = all(e < RK4_TOL_REL for e in rk4_finest)

    # ------------------------------------------------------------------ #
    # 3. linear scale invariance + mirror symmetry (analytic route)
    # ------------------------------------------------------------------ #
    a10, a20 = analytic_zigzag(model, 10.0), analytic_zigzag(model, 20.0)
    scale_rows = []
    for i in range(2):
        scale_rows.append(
            {
                "quantity": f"overshoot_{i + 1}_deg",
                "10/10": a10.overshoots_deg[i],
                "20/20": a20.overshoots_deg[i],
                "expected_20/20 = 2x 10/10": 2.0 * a10.overshoots_deg[i],
                "rel_err": rel_err(a20.overshoots_deg[i], 2.0 * a10.overshoots_deg[i]),
            }
        )
    time_rel = max(
        rel_err(x, y) for x, y in zip(a20.reversal_times + a20.peak_times, a10.reversal_times + a10.peak_times)
    )
    stbd = simulate_zigzag(model, 10.0, dt=0.1)
    port = simulate_zigzag(model, -10.0, dt=0.1)
    mirror_rel = max(
        rel_err(p, s) for p, s in zip(port.overshoots_deg, stbd.overshoots_deg)
    )
    results["invariance"] = {
        "rows": scale_rows,
        "switch_time_rel_err": time_rel,
        "mirror_overshoot_rel_err": mirror_rel,
        "max_rel_err": max(max(r["rel_err"] for r in scale_rows), time_rel, mirror_rel),
        "note": "linear model: psi(t) = delta * g(t), so 20/20 is the same solution as 10/10",
    }
    invariance_ok = results["invariance"]["max_rel_err"] < SCALE_TOL_REL

    # ------------------------------------------------------------------ #
    # 4. cross-flow nonlinearity: invariance breaks, convergence still holds
    # ------------------------------------------------------------------ #
    mn = nonlin_model(1.0)
    n10 = simulate_zigzag(mn, 10.0, dt=0.1)
    n20 = simulate_zigzag(mn, 20.0, dt=0.1)
    ratio = n20.first_overshoot_deg / n10.first_overshoot_deg
    nl_dts = (0.4, 0.2, 0.1, 0.05)
    nl_vals = [simulate_zigzag(mn, 20.0, dt=dt).first_overshoot_deg for dt in nl_dts]
    nl_fine = nl_vals[-1]
    nl_errs = [abs(v - nl_fine) for v in nl_vals[:-1]]
    results["crossflow"] = {
        "Y_vv": mn.Y_vv,
        "N_vv": mn.N_vv,
        "overshoot_10_10_deg": n10.overshoots_deg,
        "overshoot_20_20_deg": n20.overshoots_deg,
        "ratio_20_20_over_10_10_first": ratio,
        "linear_ratio": 2.0,
        "dt": list(nl_dts),
        "first_overshoot_deg": nl_vals,
        "richardson_observed_order": observed_order(nl_dts[:-1], nl_errs),
        "analytic_route_refuses": True,
    }
    inv_breaks = 1.5 < ratio < 1.99

    # ------------------------------------------------------------------ #
    # 5. IMO MSC.137(76) gate: nominal + yaw-damping degradation family
    # ------------------------------------------------------------------ #
    imo_family = []
    for scale in (1.0, 0.8, 0.7, 0.5, 0.4, 0.3):
        p = dict(NOMINAL_TANKER_PRIME)
        p["N_r"] *= scale
        m = build_model(prime=p)
        zz = analytic_zigzag(m, 20.0)
        verdict = check_imo_msc137_zigzag(zz, L, U)
        turn = m.steady_turn(math.radians(20.0))
        imo_family.append(
            {
                "N_r_scale": scale,
                "first_overshoot_deg": zz.first_overshoot_deg,
                "second_overshoot_deg": zz.second_overshoot_deg,
                "R_over_L": turn.R_over_L,
                "directionally_stable": turn.stable,
                "imo_pass": verdict["pass"],
                "imo_fail_margin_deg": min(c["margin_deg"] for c in verdict["checks"]),
            }
        )
    nominal_pass = next(r for r in imo_family if r["N_r_scale"] == 1.0)["imo_pass"]
    any_fail = any(r["imo_pass"] is False for r in imo_family)
    results["imo"] = {
        "limits_20_20": imo_msc137_zigzag_limits(20.0, 20.0, L_OVER_V)["limits_deg"],
        "limits_10_10": imo_msc137_zigzag_limits(10.0, 10.0, L_OVER_V)["limits_deg"],
        "family": imo_family,
        "nominal_pass": nominal_pass,
        "some_member_fails": any_fail,
        "degradation_note": (
            "N_r_scaled family is a sensitivity scan, not a set of physical ships: "
            "members with N_r_scale <= 0.5 give R/L < 1 and are used only to show the "
            "criterion discriminates"
        ),
    }
    imo_ok = bool(nominal_pass) and any_fail

    # ------------------------------------------------------------------ #
    # verdict
    # ------------------------------------------------------------------ #
    verdict = {
        "route_max_rel_err": route_max,
        "route_pass": route_max < ROUTE_TOL_REL,
        "euler_observed_order": euler_orders,
        "euler_first_order_pass": order_ok,
        "rk4_finest_rel_err": rk4_finest,
        "rk4_pass": rk4_ok,
        "invariance_max_rel_err": results["invariance"]["max_rel_err"],
        "invariance_pass": invariance_ok,
        "crossflow_breaks_invariance": inv_breaks,
        "imo_gate_discriminates": imo_ok,
    }
    verdict["pass"] = bool(
        verdict["route_pass"]
        and verdict["euler_first_order_pass"]
        and verdict["rk4_pass"]
        and verdict["invariance_pass"]
        and inv_breaks
        and imo_ok
    )
    results["verdict"] = verdict

    # ------------------------------------------------------------------ #
    # figures
    # ------------------------------------------------------------------ #
    make_figures(model, results, mn)

    (_HERE / "result.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
    (results["figures"] if False else _HERE / "figures").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # console summary
    # ------------------------------------------------------------------ #
    print(f"ship: KVLCC2-class tanker, L/V = {L_OVER_V:.2f} s")
    print("\n[1] two routes (reference dt = %.3f s, RK4)" % REFERENCE_DT)
    for key, block in results["routes"].items():
        print(f"    {key}: max relative difference over 6 quantities = {block['max_rel_err']:.3e}")
        for r in block["rows"]:
            print(f"        {r['quantity']:<20} sim={r['sim']:.9f} analytic={r['analytic']:.9f}"
                  f" rel={r['rel_err']:.3e}")
    print("\n[2] time-step convergence of first overshoot")
    for key, c in results["convergence"].items():
        print(f"    {key}: analytic={c['analytic_first_overshoot_deg']:.9f}")
        for name in ("semi_implicit_euler", "rk4"):
            print(f"        {name:<20} order={c[name]['observed_order']:+.3f}"
                  f" finest rel={c[name]['rel_err'][-1]:.3e}")
    print("\n[3] linear-model invariance")
    for r in results["invariance"]["rows"]:
        print(f"    {r['quantity']}: 10/10={r['10/10']:.9f} 20/20={r['20/20']:.9f}"
              f" rel={r['rel_err']:.3e}")
    print(f"    switch times rel={results['invariance']['switch_time_rel_err']:.3e}"
          f" mirror rel={results['invariance']['mirror_overshoot_rel_err']:.3e}")
    print("\n[4] cross-flow case (representative Y_vv/N_vv)")
    print(f"    10/10 first={results['crossflow']['overshoot_10_10_deg'][0]:.4f}"
          f" 20/20 first={results['crossflow']['overshoot_20_20_deg'][0]:.4f}"
          f" ratio={ratio:.4f} (linear would be 2.0)")
    print("\n[5] IMO MSC.137(76) gate, 20/20")
    for r in imo_family:
        print(f"    N_r x{r['N_r_scale']:.1f}: first={r['first_overshoot_deg']:7.3f} deg"
              f" second={r['second_overshoot_deg']:8.3f} deg R/L={r['R_over_L']:6.2f}"
              f" pass={r['imo_pass']}")
    print("\nVERDICT:", "PASS" if verdict["pass"] else "FAIL")
    for k, v in verdict.items():
        if k != "pass":
            print(f"    {k}: {v}")
    return 0 if verdict["pass"] else 1


def make_figures(model, results, nonlin_model_) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)

    # ---- fig 1: time histories, analytic vs numerical -------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, delta in enumerate(TESTS_DEG):
        ana = analytic_zigzag(model, delta)
        sim = simulate_zigzag(model, delta, dt=REFERENCE_DT, integrator="rk4")
        ax = axes[0, col]
        ax.plot(ana.history["t"], ana.history["psi_deg"], "-", lw=2, label="heading (analytic)")
        ax.plot(sim.history["t"], sim.history["psi_deg"], "--", lw=1.2, label=f"heading (RK4 dt={REFERENCE_DT})")
        for i, (tr, ov) in enumerate(zip(ana.reversal_times, ana.overshoots_deg)):
            ax.axvline(tr, color="0.5", lw=0.8, ls=":")
            ax.annotate(f"ovs{i + 1}={ov:.3f}$^\\circ$", (tr, ana.peak_times and 0), xytext=(tr + 8, delta * 1.15))
        ax.axhline(delta, color="0.7", lw=0.8, ls="--")
        ax.axhline(-delta, color="0.7", lw=0.8, ls="--")
        ax.set_ylabel("heading deviation [deg]")
        ax.set_title(f"{delta:.0f}$^\\circ$/{delta:.0f}$^\\circ$ zig-zag")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        ax.step(ana.history["t"], ana.history["rudder_deg"], "-", lw=1.5, where="post", label="rudder order")
        ax.step(sim.history["t"], sim.history["rudder_deg"], "--", lw=1.0, where="post", label="RK4")
        ax.set_ylabel("rudder angle [deg]")
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Zig-zag manoeuvre: event-driven RK4 vs exact piecewise-modal solution")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_zigzag_histories.png", dpi=130)
    plt.close(fig)

    # ---- fig 2: convergence of the overshoot angle ---------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, delta in enumerate(TESTS_DEG):
        c = results["convergence"][f"{delta:.0f}/{delta:.0f}"]
        ax = axes[col]
        for name, style in (("semi_implicit_euler", "o-"), ("rk4", "s-")):
            d = c[name]
            ax.loglog(d["dt"], d["abs_err_deg"], style, label=f"{name} (p={d['observed_order']:.2f})")
        ref = np.asarray(c["semi_implicit_euler"]["dt"], dtype=float)
        scale = c["semi_implicit_euler"]["abs_err_deg"][0] / ref[0]
        ax.loglog(ref, scale * ref, "k:", lw=1, label="slope 1 guide")
        ax.set_xlabel("time step dt [s]")
        ax.set_ylabel("|first overshoot error| [deg]")
        ax.set_title(f"{delta:.0f}$^\\circ$/{delta:.0f}$^\\circ$: convergence to the analytic value")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_overshoot_convergence.png", dpi=130)
    plt.close(fig)

    # ---- fig 3: IMO gate vs yaw damping --------------------------------- #
    fam = results["imo"]["family"]
    xs = [r["N_r_scale"] for r in fam]
    first = [r["first_overshoot_deg"] for r in fam]
    second = [r["second_overshoot_deg"] for r in fam]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, first, "o-", label="first overshoot (20/20)")
    ax.plot(xs, second, "s-", label="second overshoot (20/20)")
    ax.axhline(25.0, color="tab:red", ls="--", lw=1.2, label="IMO MSC.137(76): first <= 25$^\\circ$")
    ax.axhline(40.0, color="darkred", ls=":", lw=1.2, label="IMO MSC.137(76): second <= 40$^\\circ$")
    for r in fam:
        ax.annotate("pass" if r["imo_pass"] else "FAIL", (r["N_r_scale"], r["first_overshoot_deg"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=7,
                    color="tab:green" if r["imo_pass"] else "tab:red")
    ax.invert_xaxis()
    ax.set_xlabel("yaw-damping derivative scale  $N'_r \\times$ scale  (1.0 = nominal tanker)")
    ax.set_ylabel("overshoot angle [deg]")
    ax.set_title("IMO overshoot gate: nominal ship vs degraded course-keeping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_imo_gate.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())