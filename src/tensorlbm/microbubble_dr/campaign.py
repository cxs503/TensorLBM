"""P106W2-N01 campaign driver: microbubble drag reduction in a turbulent channel.

Two-stage driver so that the expensive runs can be spread over the cluster:

``--only NAME``   run one case (single-phase baseline, gas-fraction scan, bubble
                  diameter scan, negative controls) and write ``case_NAME.json``.
``--merge``       read all case JSONs, run the FIK mechanism decomposition,
                  evaluate the four machine criteria, write ``campaign.json``
                  and the figures.

Usage::

    PYTHONPATH=src python3 -m tensorlbm.microbubble_dr.campaign --out artifacts/microbubble_dr_p106w2_n01 --only baseline
    PYTHONPATH=src python3 -m tensorlbm.microbubble_dr.campaign --out artifacts/microbubble_dr_p106w2_n01 --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .channel import ChannelConfig, probe_coexistence, run_case
from .fik import Profile, fik_decomposition, momentum_consistency, residual_stress

#: Dean (1978) correlation, used only to *initialise* the drive (never as a
#: result): Cf = 0.073 Re_b^-0.25.
DEAN_A, DEAN_B = 0.073, 0.25


def cf_dean(re_b: float) -> float:
    return DEAN_A * re_b ** (-DEAN_B)


def base_config() -> dict:
    """Common single-phase state; every case shares ``U_target`` and ``nu``."""
    return dict(
        nx=56,
        ny=40,
        nz=24,
        Re_tau=180.0,
        u_tau=0.006,
        U_target=0.108,
        rho_l=1.38,
        rho_g=0.30,
        G=4.3,   # measured: stable two-phase at tau~0.5024 (G=4.5 diverges)
        g_b=-2.0e-4,
        lam_trt=None,
        n_warmup=1500,
        n_steps=4500,
        sample_every=20,
        pump_gain=0.05,
        seed=1234,
    )


def cases() -> dict[str, dict]:
    b = base_config()
    out: dict[str, dict] = {}
    # single-phase reference (SC switched off entirely)
    out["baseline"] = {**b, "tag": "baseline", "G": 0.0, "alpha": 0.0}
    # negative control 1: the SC model active, but no gas at all
    # rho_l = 1.45 sits just above the measured coexistence liquid density, so the
    # liquid is single-phase stable and no vapour can nucleate: gas -> 0 exactly.
    out["neg_no_gas"] = {**b, "tag": "neg_no_gas", "alpha": 0.0, "rho_l": 1.45}
    # gas-fraction scan at fixed bubble diameter (d+ = 60 -> 8 lattice cells)
    for a in (0.003, 0.01, 0.03):
        out[f"alpha_{a:g}"] = {**b, "tag": f"alpha_{a:g}", "alpha": a, "d_plus": 60.0}
    # bubble-diameter scan at MATCHED BUBBLE COUNT (one bubble): the gas volume
    # fraction is quantised by the domain (alpha = pi d^3/6 / V), so matching the
    # gas *volume* is impossible at d+ = 180; matching the bubble count keeps the
    # comparison controlled and the achieved alpha is reported per case.
    out["dplus_120"] = {**b, "tag": "dplus_120", "alpha": 0.023, "d_plus": 120.0}
    out["dplus_180"] = {**b, "tag": "dplus_180", "alpha": 0.078, "d_plus": 180.0}
    return out


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_one(name: str, outdir: Path, overrides: dict | None = None) -> dict:
    params = cases()[name]
    if overrides:
        params.update({k: v for k, v in overrides.items() if k in params})
    cfg = ChannelConfig(**params)
    rec = run_case(cfg, verbose=os.environ.get("MBDR_VERBOSE", "1") != "0")
    rec["name"] = name
    path = ensure_dir(outdir) / f"case_{name}.json"
    payload = json.loads(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    payload["config"] = params
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _add_provenance(d: dict, files: list[Path]) -> dict:
    d["provenance"] = {
        "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.exists()},
        "generator": "tensorlbm.microbubble_dr.campaign",
    }
    return d


def merge(outdir: Path) -> dict:
    """Decompose all cases, evaluate the criteria, write campaign.json + figures."""
    outdir = ensure_dir(outdir)
    recs: dict[str, dict] = {}
    for f in sorted(outdir.glob("case_*.json")):
        r = json.loads(f.read_text())
        recs[r["name"]] = r
    if "baseline" not in recs:
        raise SystemExit("baseline case missing -- run --only baseline first")
    base = recs["baseline"]

    def prof(r: dict) -> Profile:
        return Profile(
            y=np.asarray(r["y"]),
            rho=np.asarray(r["rho"]),
            u=np.asarray(r["u"]),
            uv=np.asarray(r["uv"]),
            alpha=np.asarray(r["alpha"]),
            U_b=r["U_b"],
            F0=r["F0"],
            H=r["H"],
            nu=r["nu"],
            rho_b=r["rho_b"],
        )

    base_p = prof(base)
    out: dict = {
        "node": "P106W2-N01",
        "repo": "TensorLBM",
        "module": "tensorlbm.microbubble_dr",
        "cases": {k: {kk: v[kk] for kk in ("U_b", "U_target", "F0", "Re_tau_meas", "u_tau_meas",
                                           "alpha_integral", "alpha_integral_drift_rel",
                                           "mass_drift_rel", "n_samples", "runtime_s",
                                           "d_plus_actual", "wall_units")}
                  for k, v in recs.items()},
        "criteria": {},
        "mechanisms": {},
        "profiles": {},
    }

    # ---- baseline momentum consistency (three-way u_tau) --------------------
    mc_base = momentum_consistency(base)
    out["criteria"]["momentum_baseline"] = {
        "u_tau_grad": mc_base["u_tau_grad"],
        "u_tau_reynolds": mc_base["u_tau_reynolds"],
        "u_tau_force": mc_base["u_tau_force"],
        "u_tau_mem": mc_base["u_tau_mem_bottom"],
        "spread_rel": mc_base["u_tau_spread_rel"],
        "threshold": 0.02,
        "pass": bool(mc_base["u_tau_spread_rel"] <= 0.02),
    }

    # ---- per-case decomposition --------------------------------------------
    for name, r in recs.items():
        if name == "baseline":
            continue
        p = prof(r)
        C_guess = r["F0"] * r["H"] / 2.0      # first guess for tau_b extraction
        tb = residual_stress(p.y, p.u, p.rho, p.uv, p.nu, C_guess, p.F0)
        d = fik_decomposition(p, base_p, tau_b=tb)
        mc = momentum_consistency(r)
        # the FIK wall shear vs the independent momentum-exchange measurement
        closure = abs(d["tau_top"] - r["mem_top"]) / max(r["mem_top"], 1e-30)
        out["mechanisms"][name] = {
            "DR": d["DR"],
            "DR_visc": d["DR_visc"],
            "DR_turb": d["DR_turb"],
            "DR_slip": d["DR_slip"],
            "dCf": d["dCf"],
            "dCf_visc": d["dCf_visc"],
            "dCf_turb": d["dCf_turb"],
            "dCf_slip": d["dCf_slip"],
            "tau_top_fik": d["tau_top"],
            "tau_top_mem": r["mem_top"],
            "fik_vs_mem_rel": closure,
            "sum_residual_rel": d["sum_residual_rel"],
            "criterion_sum_pass": bool(
                abs(d["dCf_visc"] + d["dCf_turb"] + d["dCf_slip"] - d["dCf"])
                <= 0.03 * abs(d["dCf"]) + 1e-12
            ),
            "criterion_fik_mem_pass": bool(closure <= 0.03),
            "momentum": mc,
            "alpha_integral_drift_rel": r["alpha_integral_drift_rel"],
            "mass_drift_rel": r["mass_drift_rel"],
        }
        out["profiles"][name] = {
            "y": r["y"],
            "u": r["u"],
            "rho": r["rho"],
            "uv": r["uv"],
            "alpha": r["alpha"],
            "tau_b": tb.tolist(),
        }

    # ---- criteria ----------------------------------------------------------
    ms = out["mechanisms"]
    neg = ms.get("neg_no_gas", {})
    out["criteria"]["negative_control_zero_gas"] = {
        "DR": neg.get("DR"),
        "dCf_over_Cf": neg.get("DR"),
        "threshold": 0.005,
        "pass": bool(abs(neg.get("DR", 1.0)) <= 0.005),
    }
    alpha_cases = {k: v for k, v in ms.items() if k.startswith("alpha_")}
    out["criteria"]["mass_conservation"] = {
        name: {
            "alpha_integral_drift_rel": v["alpha_integral_drift_rel"],
            "mass_drift_rel": v["mass_drift_rel"],
            "pass": bool(v["alpha_integral_drift_rel"] <= 0.01),
        }
        for name, v in alpha_cases.items()
    }
    out["criteria"]["mechanism_sum"] = {
        name: {
            "sum_residual_rel": v["sum_residual_rel"],
            "fik_vs_mem_rel": v["fik_vs_mem_rel"],
            "pass": bool(v["criterion_sum_pass"] and v["criterion_fik_mem_pass"]),
        }
        for name, v in ms.items()
    }
    # diameter saturation trend
    dsw = {k: ms[k] for k in ("alpha_0.003", "dplus_120", "dplus_180") if k in ms}
    if dsw:
        order = sorted(dsw, key=lambda k: recs[k]["d_plus_actual"])
        out["criteria"]["diameter_trend"] = {
            "sequence": order,
            "note": "matched bubble count (one bubble); alpha grows with d^3 by construction",
            "d_plus": [recs[k]["d_plus_actual"] for k in order],
            "alpha": [recs[k]["alpha_integral"] for k in order],
            "DR": [dsw[k]["DR"] for k in order],
            "DR_over_alpha": [dsw[k]["DR"] / max(recs[k]["alpha_integral"], 1e-12) for k in order],
            "monotone_saturating": bool(
                all(dsw[order[i + 1]]["DR"] <= dsw[order[i]]["DR"] + 0.02 for i in range(len(order) - 1))
            ),
        }
    _add_provenance(out, list(outdir.glob("case_*.json")))
    (outdir / "campaign.json").write_text(json.dumps(out, indent=2))
    try:
        make_figures(out, outdir)
    except Exception as exc:  # pragma: no cover - plotting must not eat the result
        out["figure_error"] = repr(exc)
        (outdir / "campaign.json").write_text(json.dumps(out, indent=2))
    return out


def make_figures(out: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    prof = out["profiles"]
    base = prof.get("baseline")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    if base is not None:
        y = np.asarray(base["y"])
        H = float(y[-1] + 0.5)
        for name, p in prof.items():
            yy = np.asarray(p["y"])
            ax[0].plot(np.asarray(p["u"]), yy / H, label=name)
            ax[1].plot(-np.asarray(p["uv"]), yy / H, label=name)
            ax[2].plot(np.asarray(p["alpha"]), yy / H, label=name)
    ax[0].set_xlabel(r"$\bar u$ [lattice]")
    ax[0].set_ylabel(r"$y/H$")
    ax[0].set_title("mean velocity")
    ax[1].set_xlabel(r"$-\langle u'v'\rangle$")
    ax[1].set_title("Reynolds shear stress")
    ax[2].set_xlabel(r"$\alpha$")
    ax[2].set_title("gas volume fraction")
    for a in ax:
        a.legend(fontsize=6)
    fig.tight_layout()
    f1 = outdir / "fig1_profiles.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    figs.append(f1.name)

    ms = out["mechanisms"]
    names = list(ms)
    if names:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        visc = [ms[n]["DR_visc"] for n in names]
        turb = [ms[n]["DR_turb"] for n in names]
        slip = [ms[n]["DR_slip"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, visc, label="viscosity")
        ax.bar(x, turb, bottom=np.array(visc), label="turbulence modulation")
        ax.bar(x, slip, bottom=np.array(visc) + np.array(turb), label="phase slip")
        ax.plot(x, [ms[n]["DR"] for n in names], "k*", label="total (FIK)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(r"$\Delta C_f/C_{f0}$")
        ax.legend(fontsize=7)
        fig.tight_layout()
        f2 = outdir / "fig2_mechanism_split.png"
        fig.savefig(f2, dpi=140)
        plt.close(fig)
        figs.append(f2.name)
    return figs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    if args.list:
        print("\n".join(cases()))
        return 0
    if args.probe:
        probe = probe_coexistence(ChannelConfig(**base_config()))
        probe["config"] = base_config()
        ensure_dir(outdir)
        (outdir / "coexistence.json").write_text(json.dumps(probe, indent=2))
        print(json.dumps(probe, indent=2))
        return 0
    if args.merge:
        res = merge(outdir)
        compact = {
            "criteria": res["criteria"],
            "mechanisms": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                           for k, v in res["mechanisms"].items()},
            "cases": res["cases"],
            "profiles": {k: {kk: [round(float(x), 6) for x in np.asarray(p[kk]).ravel()]
                             for kk in ("y", "u", "uv", "alpha", "rho")}
                         for k, p in res["profiles"].items()},
        }
        print("COMPACT_JSON_BEGIN")
        print(json.dumps(compact))
        print("COMPACT_JSON_END")
        return 0
    if not args.only:
        raise SystemExit("need --only NAME or --merge")
    rec = run_one(args.only, outdir)
    print(json.dumps({k: rec[k] for k in ("name", "U_b", "F0", "Re_tau_meas", "alpha_integral",
                                          "mass_drift_rel", "runtime_s")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())