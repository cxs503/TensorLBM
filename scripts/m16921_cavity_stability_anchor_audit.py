"""m16921 — anchor audit of the cavity linear-stability module.

Produces ``artifacts/m16921_cavity_stability_anchor_audit.json`` (every number in
the report comes from here, no hand copying) and the two figures used by
``docs/M16921_CAVITY_STABILITY_ANCHOR_AUDIT.md``.

What it establishes
-------------------
1. ``conjugate_thermal.stability.critical_rayleigh`` (side-heated square cavity)
   is **stable** at the classical Rayleigh--Bénard anchor ``Ra = 1707.76``; its
   own neutral point is an order of magnitude higher.  The module used to claim
   it reproduced ``1707.762`` "by an independent discretisation".
2. The newly added Bénard path (gravity parallel to the temperature gradient)
   does reproduce ``1707.7597``, second order, with Richardson extrapolation.
3. The Bénard growth rates agree with the independent streamfunction-pencil
   implementation ``rb_continuum_stability`` once the ``1/Pr`` time-unit
   conversion is applied.

Run: ``PYTHONPATH=src python scripts/m16921_cavity_stability_anchor_audit.py``
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tensorlbm.conjugate_thermal import stability as S
import tensorlbm.rb_continuum_stability as RB

OUT = Path(__file__).resolve().parents[1] / "artifacts"
RA_REF = S.RA_C_BENARD_RIGID_RIGID
A_REF = S.A_C_BENARD_RIGID_RIGID
PR = 0.71


def main() -> dict:
    t_start = time.time()
    res: dict = {
        "mission": "m16921",
        "reference": {"Ra_c": RA_REF, "a_c": A_REF, "Pr": PR,
                      "source": "Chandrasekhar 1961 §II, rigid-rigid Bénard layer"},
        "timing": {},
    }

    # ---- 1. the old claim, tested at its default arguments ----------------
    t0 = time.time()
    legacy = {}
    for n in (8, 12, 16):
        r = S.critical_rayleigh(n=n)
        legacy[f"n{n}"] = {
            "ra_c": r["ra_c"],
            "status": r["status"],
            "deviation_rel": r["deviation_rel"],
            "sigma_at_bracket": list(r["sigma_bracket"]),
            "has_deviation_rel_key": "deviation_rel" in r,
        }
    res["legacy_side_heated_default_call"] = legacy
    res["timing"]["legacy"] = round(time.time() - t0, 2)

    # ---- 2. side-heated sigma(Ra) map (n = 12) ---------------------------
    t0 = time.time()
    ra_map = [500.0, 1000.0, RA_REF, 5000.0, 1.0e4, 1.45e4, 3.0e4]
    side = {f"{ra:g}": S.growth_rate(ra, n=12) for ra in ra_map}
    res["side_heated_sigma"] = {"n": 12, "pr": PR, "sigma_vs_ra": side}
    sh = S.critical_rayleigh(n=12, ra_lo=1.0e4, ra_hi=3.0e4, tol=1e-3)
    res["side_heated_neutral"] = {
        "n": 12, "ra_c": sh["ra_c"], "status": sh["status"],
        "ratio_to_benard_anchor": (None if sh["ra_c"] is None
                                   else sh["ra_c"] / RA_REF),
    }
    res["timing"]["side_heated"] = round(time.time() - t0, 2)

    # ---- 3. Bénard path: grid convergence + Richardson -------------------
    t0 = time.time()
    benard_ra_c = {}
    for n in (16, 32, 64):
        r = S.benard_critical_rayleigh(n=n, a=A_REF)
        benard_ra_c[f"n{n}"] = {"ra_c": r["ra_c"], "deviation_rel": r["deviation_rel"],
                                "sigma_marginal": r.get("sigma_marginal"),
                                "status": r["status"]}
    vals = [benard_ra_c[f"n{n}"]["ra_c"] for n in (16, 32, 64)]
    extrap = S.richardson(vals)
    res["benard_grid_convergence"] = {
        "per_grid": benard_ra_c,
        "richardson_16_32_64": extrap,
        "richardson_rel_err": extrap / RA_REF - 1.0,
        "error_ratios": [abs(benard_ra_c["n16"]["deviation_rel"]
                             / benard_ra_c["n32"]["deviation_rel"]),
                         abs(benard_ra_c["n32"]["deviation_rel"]
                             / benard_ra_c["n64"]["deviation_rel"])],
    }
    res["timing"]["benard"] = round(time.time() - t0, 2)

    # ---- 4. wavenumber scan ---------------------------------------------
    t0 = time.time()
    a_list = [2.8, 3.0, A_REF, 3.25, 3.5]
    scan = S.benard_wavenumber_scan(a_list, n=32)
    res["benard_wavenumber_scan_n32"] = scan
    best = min(scan, key=lambda g: g["Ra_c"])
    res["benard_most_dangerous_wavenumber"] = {"a": best["a"], "Ra_c": best["Ra_c"],
                                               "rel_err_vs_ref_a": best["a"] / A_REF - 1.0}
    res["timing"]["wavenumber"] = round(time.time() - t0, 2)

    # ---- 5. cross-check vs the independent streamfunction pencil --------
    t0 = time.time()
    cross = []
    for ra in (1500.0, 2500.0, 3500.0):
        here = S.benard_growth_rate(ra, n=64, a=A_REF, pr=PR)
        there = RB.growth(a=A_REF, Ra=ra, Pr=PR, M=100)["sigma"]
        cross.append({"Ra": ra, "sigma_thermal_units_n64": here,
                      "sigma_viscous_units_M100": there,
                      "pr_times_rb": PR * there,
                      "rel_diff_after_conversion": PR * there / here - 1.0,
                      "rel_diff_naive": there / here - 1.0})
    res["cross_check_rb_continuum"] = {"pr": PR, "rows": cross}
    res["timing"]["cross_check"] = round(time.time() - t0, 2)
    res["timing"]["total"] = round(time.time() - t_start, 2)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m16921_cavity_stability_anchor_audit.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    return res


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUT / "m16921_cavity_stability_anchor_audit.json").read_text())
    RA_REF = data["reference"]["Ra_c"]

    # --- figure 1: sigma(Ra) for both configurations ----------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    side = data["side_heated_sigma"]["sigma_vs_ra"]
    ra_s = np.array([float(k) for k in side])
    sg_s = np.array([side[k] for k in side])
    ax.plot(ra_s, sg_s, "s-", color="#c0392b",
            label="side-heated square cavity (n=12)")
    ben = []
    for ra in (500.0, 1000.0, RA_REF, 2500.0, 5000.0):
        ben.append((ra, S.benard_growth_rate(ra, n=32, a=A_REF, pr=PR)))
    ra_b = np.array([b[0] for b in ben])
    sg_b = np.array([b[1] for b in ben])
    ax.plot(ra_b, sg_b, "o-", color="#1f4e79",
            label="Rayleigh–Bénard layer (n=32, a=a_c)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(RA_REF, color="0.4", ls="--", lw=1.0)
    ax.text(RA_REF * 1.05, ax.get_ylim()[0] * 0.55, "Chandrasekhar\nRa$_c$=1707.76",
            fontsize=8, color="0.3")
    sh_ra = data["side_heated_neutral"]["ra_c"]
    ax.axvline(sh_ra, color="#c0392b", ls=":", lw=1.0)
    ax.text(sh_ra * 1.05, ax.get_ylim()[1] * 0.25,
            f"side-heated\nRa$_c$≈{sh_ra:.2e}", fontsize=8, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xlabel("Rayleigh number Ra")
    ax.set_ylabel(r"growth rate $\sigma$  [$\alpha/H^2$]")
    ax.set_title("Same module, two configurations: the 1707.76 anchor\n"
                 "belongs to the Bénard layer only", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "m16921_sigma_vs_ra_configurations.png", dpi=160)
    plt.close(fig)

    # --- figure 2: grid convergence of the Bénard Ra_c --------------------
    per = data["benard_grid_convergence"]["per_grid"]
    ns = [int(k[1:]) for k in per]
    hs2 = [(1.0 / n) ** 2 for n in ns]
    ras = [per[k]["ra_c"] for k in per]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(hs2, ras, "o-", color="#1f4e79", label=r"$Ra_c$ (vorticity/Thom FD)")
    ax.axhline(RA_REF, color="#c0392b", ls="--",
               label=f"Chandrasekhar {RA_REF}")
    ext = data["benard_grid_convergence"]["richardson_16_32_64"]
    ax.plot([0.0], [ext], "*", ms=13, color="#27ae60",
            label=f"Richardson: {ext:.3f}\n({1e2*(ext/RA_REF-1):+.4f}%)")
    for h2, ra, n in zip(hs2, ras, ns):
        ax.annotate(f"n={n}", (h2, ra), textcoords="offset points",
                    xytext=(5, -10), fontsize=8)
    ax.set_xlabel(r"$h^2$  (h = 1/n)")
    ax.set_ylabel(r"critical Rayleigh number $Ra_c$")
    ax.set_title("Bénard layer: second-order convergence onto 1707.7597", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "m16921_benard_grid_convergence.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
    figures()
    print("done")