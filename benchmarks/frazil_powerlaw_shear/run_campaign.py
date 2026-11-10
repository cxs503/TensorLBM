#!/usr/bin/env python3
"""Frazil power-law rheology campaign: structure -> LBM shear cell -> (K, n).

Usage
-----
::

    python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --plan
    python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --run --shard 0 --shards 16 --threads 2
    python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --assemble
    python3 benchmarks/frazil_powerlaw_shear/run_campaign.py --figure

``--plan`` writes ``cases.json`` (deterministic, no simulation).  ``--run``
executes the cases whose index satisfies ``idx % shards == shard`` and writes one
JSON per case, so the campaign parallelises over independent processes/nodes and
a crashed case loses only itself.  ``--assemble`` merges the case files into
``artifacts/frazil_powerlaw_shear/campaign.json`` (with the five acceptance
criteria evaluated in the artifact, never in the prose).  ``--figure`` renders
the figures from the artifact.

Units: lattice.  ``nu = 0.2``, ``rho = 1``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Campaign definition (declared before any run -- no post-hoc choices)
# ---------------------------------------------------------------------------

NU = 0.2                     # lattice kinematic viscosity, rho = 1
GAMMA_RATES = (5.106e-4, 1.0212e-3, 2.0424e-3)   # factor-2 ladder, 2 decades/4
DISPERSED_PHI = (0.05, 0.10, 0.20, 0.30)
FLOCCULATED_PHI = (0.05, 0.10)
N_CAP = 8                    # largest aggregate at the lowest shear rate
DF = 2.0

#: Two scales of the same campaign.  ``production`` is the one the report quotes
#: when the cluster is available; ``reduced`` is the same physics at a smaller
#: cell (32^3 / 40^3 instead of 48^3 / 64^3) so the campaign can also be run on a
#: contended single host in about half an hour.  The scale actually used is
#: recorded inside the artifact.
SCALES = {
    "production": {
        "c48": dict(nx=48, ny=48, nz=49, radius=3.0, wall_margin=4, n_steps=3600,
                    n_warmup=2400, sample_every=20),
        "f64": dict(nx=64, ny=64, nz=65, radius=4.0, wall_margin=5, n_steps=5200,
                    n_warmup=3600, sample_every=25),
    },
    "reduced": {
        "c32": dict(nx=32, ny=32, nz=33, radius=2.5, wall_margin=3, n_steps=1200,
                    n_warmup=800, sample_every=10),
        "f40": dict(nx=40, ny=40, nz=41, radius=3.0, wall_margin=3, n_steps=1600,
                    n_warmup=1050, sample_every=10),
    },
}
SCALE = os.environ.get("FRAZIL_SCALE", "production")
RESOLUTIONS = SCALES[SCALE]
COARSE_TAG, FINE_TAG = list(RESOLUTIONS)[0], list(RESOLUTIONS)[1]

#: resolution pair used for the grid-convergence criterion, per structure family
CONVERGENCE = {
    "flocculated_c0.10": (COARSE_TAG, FINE_TAG, "flocculated", 0.10),
    "dispersed_c0.20": (COARSE_TAG, FINE_TAG, "dispersed", 0.20),
}


def _anchor_bond_force() -> dict:
    """Anchor ``F_b`` so that the largest floc (``N_CAP``) sits at the lowest rate."""
    from tensorlbm.frazil_powerlaw_shear import bond_force_from_reference

    g_ref = min(GAMMA_RATES)
    a = RESOLUTIONS[COARSE_TAG]["radius"]
    fb = bond_force_from_reference(g_ref, a, NU, N_CAP, DF)
    return {"bond_force_lattice": fb, "gamma_ref": g_ref, "radius": a, "mu": NU,
            "n_ref": N_CAP, "df": DF}


def build_plan() -> dict:
    """Deterministic case list + the structure law's anchored parameters."""
    from tensorlbm.frazil_powerlaw_shear import equilibrium_floc_size

    anchor = _anchor_bond_force()
    anchor_per_res = {
        tag: {**_anchor_bond_force(), "radius": res["radius"],
              "bond_force_lattice": __import__("tensorlbm.frazil_powerlaw_shear", fromlist=["x"]
                                               ).bond_force_from_reference(
                  min(GAMMA_RATES), res["radius"], NU, N_CAP, DF)}
        for tag, res in RESOLUTIONS.items()}
    cases: list[dict] = []
    for tag, res in RESOLUTIONS.items():
        for gi, gamma in enumerate(GAMMA_RATES):
            u_wall = gamma * (res["nz"] - 2) / 2.0
            cases.append(dict(
                kind="pure", res=tag, family="pure", phi=0.0, gamma_target=gamma,
                u_wall=u_wall, n_primary=0, seed=1, rate_index=gi, **_res_fields(res)))
        # dispersed family (hard spheres, no aggregation)
        for phi in DISPERSED_PHI:
            if tag != COARSE_TAG and not (phi == 0.20):
                continue
            for gi, gamma in enumerate(GAMMA_RATES):
                u_wall = gamma * (res["nz"] - 2) / 2.0
                cases.append(dict(
                    kind="shear", res=tag, family="dispersed", phi=phi,
                    gamma_target=gamma, u_wall=u_wall, n_primary=1, seed=11 + gi,
                    rate_index=gi, **_res_fields(res)))
        # flocculated family (aggregate size from the shear-breakup balance)
        for phi in FLOCCULATED_PHI:
            if tag != COARSE_TAG and not (phi == 0.10):
                continue
            for gi, gamma in enumerate(GAMMA_RATES):
                n_eq = equilibrium_floc_size(
                    gamma, radius=res["radius"], mu=NU,
                    bond_force=anchor_per_res[tag]["bond_force_lattice"],
                    df=DF, n_cap=N_CAP)["n_primary"]
                u_wall = gamma * (res["nz"] - 2) / 2.0
                cases.append(dict(
                    kind="shear", res=tag, family="flocculated", phi=phi,
                    gamma_target=gamma, u_wall=u_wall, n_primary=int(n_eq), seed=31 + gi,
                    rate_index=gi, **_res_fields(res)))
    # seed replicates at the lowest rate (scatter estimate for the two criteria
    # concentrations) -- same resolution, different microstructure
    res = RESOLUTIONS[COARSE_TAG]
    gamma = GAMMA_RATES[0]
    u_wall = gamma * (res["nz"] - 2) / 2.0
    for seed in (101, 102):
        cases.append(dict(kind="shear", res=COARSE_TAG, family="dispersed", phi=0.20,
                          gamma_target=gamma, u_wall=u_wall, n_primary=1, seed=seed,
                          rate_index=0, replicate=True, **_res_fields(res)))
        cases.append(dict(kind="shear", res=COARSE_TAG, family="flocculated", phi=0.10,
                          gamma_target=gamma, u_wall=u_wall, n_primary=N_CAP, seed=seed,
                          rate_index=0, replicate=True, **_res_fields(res)))
    for i, c in enumerate(cases):
        c["index"] = i
    return {"anchor": anchor, "anchor_per_res": anchor_per_res, "cases": cases,
            "scale": SCALE, "resolutions": {k: dict(v) for k, v in RESOLUTIONS.items()}, "gamma_rates": list(GAMMA_RATES),
            "nu": NU, "df": DF, "n_cap": N_CAP,
            "concentrations": {"dispersed": list(DISPERSED_PHI),
                               "flocculated": list(FLOCCULATED_PHI)}}


def _res_fields(res: dict) -> dict:
    return {k: res[k] for k in ("nx", "ny", "nz", "radius", "wall_margin",
                                "n_steps", "n_warmup", "sample_every")}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_case(spec: dict, outdir: str, threads: int = 2) -> dict:
    import torch

    torch.set_num_threads(max(1, int(threads)))
    from tensorlbm.frazil_powerlaw_shear import pure_case, run_shear_case

    t0 = time.time()
    if spec["kind"] == "pure":
        rec = pure_case(nx=spec["nx"], ny=spec["ny"], nz=spec["nz"], nu=NU,
                        u_wall=spec["u_wall"], n_steps=spec["n_steps"],
                        n_warmup=spec["n_warmup"], sample_every=spec["sample_every"],
                        wall_band=spec["wall_margin"])
        rec["phi_lattice"] = 0.0
    else:
        rec = run_shear_case(phi=spec["phi"], u_wall=spec["u_wall"], nx=spec["nx"],
                             ny=spec["ny"], nz=spec["nz"], radius=spec["radius"], nu=NU,
                             n_primary=spec["n_primary"], seed=spec["seed"], df=DF,
                             n_steps=spec["n_steps"], n_warmup=spec["n_warmup"],
                             sample_every=spec["sample_every"], wall_margin=spec["wall_margin"],
                             placement_attempts=80000)
    rec["spec"] = spec
    rec["seconds"] = time.time() - t0
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"case_{spec['index']:03d}.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    return rec


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _load_results(outdir: str) -> list[dict]:
    recs = []
    if not os.path.isdir(outdir):
        return recs
    for name in sorted(os.listdir(outdir)):
        if name.startswith("case_") and name.endswith(".json"):
            with open(os.path.join(outdir, name)) as fh:
                recs.append(json.load(fh))
    return recs


def assemble(outdir: str, plan: dict, out_json: str) -> dict:
    from tensorlbm.frazil_powerlaw_shear import (fit_power_law, grid_convergence,
                                                evaluate_acceptance, relative_viscosity)
    recs = _load_results(outdir)
    pure = {}
    for r in recs:
        if r.get("family") == "pure":
            pure[(r["spec"]["res"], r["spec"]["rate_index"])] = r
    # group suspension runs by (res, family, phi, rate)
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        s = r.get("spec") or {}
        if s.get("kind") != "shear":
            continue
        key = (s["res"], s["family"], round(float(s["phi"]), 4), int(s["rate_index"]))
        groups.setdefault(key, []).append(r)

    per_case = []
    for key, runs in sorted(groups.items()):
        res, family, phi, gi = key
        p = pure.get((res, gi))
        if p is None:
            per_case.append({"res": res, "family": family, "phi": phi, "rate_index": gi,
                             "error": "missing pure reference"})
            continue
        etas = []
        for r in runs:
            rv = relative_viscosity(r, p)
            etas.append(rv["eta_r"])
            per_case.append({
                "res": res, "family": family, "phi": phi, "rate_index": gi,
                "seed": r["spec"]["seed"], "n_primary": r["spec"]["n_primary"],
                "gamma_dot": r["gamma_dot"], "gamma_target": r["spec"]["gamma_target"],
                "tau": r["tau"], "eta_r": rv["eta_r"], "eta_susp": rv["eta_susp"],
                "eta_pure": rv["eta_pure"], "profile_r2": r["profile_r2"],
                "re_p": r["re_p"], "phi_lattice": r.get("phi_lattice"),
                "n_aggregates": r.get("n_aggregates"),
                "mean_n_primary": r.get("mean_n_primary"),
                "rg_over_a_mean": r.get("rg_over_a_mean"),
                "df_implied_mean": r.get("df_implied_mean"),
                "hydrodynamic_phi": r.get("hydrodynamic_phi"),
                "node_overlaps": r.get("node_overlaps"),
                "seconds": r.get("seconds"),
            })
        # mean over seeds for the flow curve (a single-seed group gets its
        # aggregated row too: the curve must exist whatever the replicate count)
        ts = [r["tau"] for r in runs]
        gs = [r["gamma_dot"] for r in runs]
        if True:
            per_case.append({
                "res": res, "family": family, "phi": phi, "rate_index": gi, "seed": "mean",
                "gamma_dot": float(sum(gs) / len(gs)),
                "tau": float(sum(ts) / len(ts)),
                "eta_r": float(sum(etas) / len(etas)),
                "eta_r_spread": float(max(etas) - min(etas)) if len(etas) > 1 else 0.0,
                "profile_r2": runs[0].get("profile_r2"),
                "re_p": runs[0].get("re_p"),
                "phi_lattice": runs[0].get("phi_lattice"),
                "n_aggregates": runs[0].get("n_aggregates"),
                "mean_n_primary": runs[0].get("mean_n_primary"),
                "rg_over_a_mean": runs[0].get("rg_over_a_mean"),
                "df_implied_mean": runs[0].get("df_implied_mean"),
                "hydrodynamic_phi": runs[0].get("hydrodynamic_phi"),
                "n_seeds": len(runs), "aggregated": True,
            })

    # optional drag calibration: converts the mask volume fraction into the
    # hydrodynamic one (phi_h / phi_mask = (a_h/a_eff)^3) per resolution tag
    phi_ratio = {}
    drag_gate: dict = {}
    art_dir = os.path.dirname(out_json)
    if os.path.isdir(art_dir):
        for name in sorted(os.listdir(art_dir)):
            if name.startswith("drag_calibration") and name.endswith(".json"):
                try:
                    with open(os.path.join(art_dir, name)) as fh:
                        cal = json.load(fh)
                    tag = name.replace("drag_calibration_", "").replace(".json", "")
                    # The conversion may only be applied if the drag gate closes
                    # its own momentum budget (raw drag vs Stokes-with-confinement
                    # in *this* cell).  An unconverged or geometrically
                    # inconsistent gate is recorded and *not* used: letting it
                    # flip a criterion would be a fudge, not a correction.
                    gate_ok = float(cal.get("err_pct_corrected", 1e9)) < 5.0
                    drag_gate[tag] = {
                        "phi_ratio": float(cal["phi_ratio"]),
                        "a_h_over_a_eff": float(cal.get("a_h_over_a_eff", float("nan"))),
                        "err_pct_raw": float(cal.get("err_pct", float("nan"))),
                        "err_pct_confined": float(cal.get("err_pct_corrected", float("nan"))),
                        "applied": bool(gate_ok),
                    }
                    if gate_ok:
                        phi_ratio[tag] = float(cal["phi_ratio"])
                except Exception:  # noqa: BLE001 - a broken calibration must not hide runs
                    continue

    # flow-curve fits: one per (res, family, phi) using the seed-mean points
    curves: dict[tuple, dict] = {}
    for (res, family, phi) in sorted({(c["res"], c["family"], c["phi"]) for c in per_case}):
        pts = [c for c in per_case
               if c["res"] == res and c["family"] == family and c["phi"] == phi
               and c.get("seed") == "mean" and "tau" in c]
        if len(pts) < 2:
            continue
        fit = fit_power_law([p["gamma_dot"] for p in pts], [p["tau"] for p in pts])
        fit["eta_r_ref"] = pts[0]["eta_r"]          # lowest rate = closest to Stokes
        fit["eta_r_points"] = [p["eta_r"] for p in pts]
        fit["phi_lattice"] = pts[0].get("phi_lattice")
        if family == "dispersed" and res in phi_ratio and fit["phi_lattice"]:
            fit["phi_hydrodynamic"] = float(fit["phi_lattice"] * phi_ratio[res])
            fit["phi_ratio_applied"] = phi_ratio[res]
        fit["hydrodynamic_phi"] = pts[0].get("hydrodynamic_phi")
        fit["mean_n_primary"] = pts[0].get("mean_n_primary")
        fit["gamma_dots"] = [p["gamma_dot"] for p in pts]
        fit["taus"] = [p["tau"] for p in pts]
        curves[(res, family, phi)] = fit

    # particle-free flow curve: the c = 0 negative control (its reference is the
    # exact Couette solution tau = rho nu gdot, so n = 1 and eta_r = 1 by physics)
    pure_curves = {}
    for res in sorted({k[0] for k in pure}):
        pts = [pure[(res, gi)] for gi in sorted(g for r, g in pure if r == res)]
        if len(pts) < 2:
            continue
        pf = fit_power_law([p["gamma_dot"] for p in pts], [p["tau"] for p in pts])
        ratio = [p["tau"] / (p["gamma_dot"] * p["nu"]) for p in pts]
        pf["eta_r"] = float(sum(ratio) / len(ratio))
        pf["eta_r_spread"] = float(max(ratio) - min(ratio))
        pf["phi_lattice"] = 0.0
        pure_curves[res] = pf

    # acceptance inputs
    def _fam(family, res=COARSE_TAG):
        out = {}
        if family == "dispersed" and res in pure_curves:
            pf = pure_curves[res]
            out["0.00"] = {"n": pf["n"], "K": pf["K"], "r2": pf["r2"],
                           "eta_r": pf["eta_r"], "phi_lattice": 0.0,
                           "kd_prediction": 1.0, "source": "particle-free control"}
        for (r, f, phi), fit in curves.items():
            if r == res and f == family:
                out[f"{phi:.2f}"] = {
                    "n": fit["n"], "K": fit["K"], "r2": fit["r2"],
                    "eta_r": fit["eta_r_ref"], "phi_lattice": fit["phi_lattice"],
                    "phi_hydrodynamic": fit.get("phi_hydrodynamic"),
                    "kd_prediction": (1.0 - (fit["phi_lattice"] or phi) / 0.60) ** -1.5,
                }
        return out

    convergence = {}
    for name, (lo, hi, family, phi) in CONVERGENCE.items():
        a = curves.get((lo, family, phi))
        b = curves.get((hi, family, phi))
        if a and b:
            convergence[name] = grid_convergence(a, b)

    criteria = evaluate_acceptance(dispersed=_fam("dispersed"),
                                   flocculated=_fam("flocculated"),
                                   convergence=convergence.get("flocculated_c0.10"))

    artifact = {
        "schema": "frazil_powerlaw_shear/v1",
        "plan": plan,
        "cases_expected": len(plan["cases"]),
        "cases_found": len(recs),
        "per_case": per_case,
        "scale": plan.get("scale"),
        "flow_curves": {f"{r}|{f}|{phi:.2f}": v for (r, f, phi), v in curves.items()},
        "pure_reference": {f"{k[0]}|{k[1]}": {
            "tau": v["tau"], "gamma_dot": v["gamma_dot"], "profile_r2": v["profile_r2"],
            "nu": v["nu"], "u_wall": v["u_wall"]} for k, v in pure.items()},
        "drag_calibration_phi_ratio": phi_ratio,
        "drag_gate": drag_gate,
        "pure_flow_curves": {k: {kk: v[kk] for kk in ("n", "K", "r2", "eta_r",
                                                      "eta_r_spread", "gamma_range")}
                             for k, v in pure_curves.items()},
        "convergence": convergence,
        "acceptance": criteria,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(artifact, fh, indent=1, default=float)
    return artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--only", default=None,
                    help="comma-separated case indices to run (probe subset)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--artifact",
                    default=os.path.join(ROOT, "artifacts", "frazil_powerlaw_shear",
                                         "campaign.json"))
    args = ap.parse_args(argv)

    plan = build_plan()
    if args.plan:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(HERE, "cases.json")
        with open(path, "w") as fh:
            json.dump(plan, fh, indent=1, default=float)
        print(f"plan: {len(plan['cases'])} cases -> {path}")
        print(json.dumps(plan["anchor"], indent=1))
        return 0
    if args.run:
        todo = [c for c in plan["cases"] if c["index"] % args.shards == args.shard]
        if args.only:
            keep = {int(v) for v in str(args.only).split(",") if v.strip() != ""}
            todo = [c for c in todo if c["index"] in keep]
        print(f"shard {args.shard}/{args.shards}: {len(todo)} cases", flush=True)
        for spec in todo:
            out = os.path.join(args.outdir, f"case_{spec['index']:03d}.json")
            if os.path.exists(out):
                print(f"[{spec['index']}] cached", flush=True)
                continue
            t0 = time.time()
            try:
                run_case(spec, args.outdir, threads=args.threads)
                print(f"[{spec['index']}] {spec['family']} phi={spec['phi']} "
                      f"N={spec['n_primary']} res={spec['res']} {time.time()-t0:.1f}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - one case must not kill the shard
                with open(os.path.join(args.outdir, f"case_{spec['index']:03d}.error.json"),
                          "w") as fh:
                    json.dump({"error": repr(exc), "spec": spec}, fh, indent=1)
                print(f"[{spec['index']}] ERROR {exc!r}", flush=True)
        return 0
    if args.assemble:
        art = assemble(args.outdir, plan, args.artifact)
        print(json.dumps({"cases_found": art["cases_found"],
                          "criteria": {k: v.get("passed") for k, v in
                                       art["acceptance"]["criteria"].items()},
                          "n_pass": art["acceptance"]["n_pass"],
                          "artifact": args.artifact}, indent=1))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())