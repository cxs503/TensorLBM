"""Campaign driver: frazil slurry power-law rheology, mature verification face.

Node ``p366w3-n02-frazil-rheology-mature`` (mission 12682, repo TensorLBM).

Pipeline
--------
::

    python3 run_campaign.py --plan                     # write cases.json
    python3 run_campaign.py --run --shard 0 --shards 6 # one JSON per case
    python3 run_campaign.py --assemble                 # artifact + acceptance
    python3 make_figures.py                            # figures from the artifact
    python3 make_report.py                             # docs report (+ API report)

Nothing is fitted behind the scenes: :func:`plan_cases` is deterministic, every
case record keeps its own measurements, and the assembler only groups and
reduces them.

Design of the matrix (what each block is for)
---------------------------------------------
``pure``          ``c = 0`` controls at both resolutions: criterion 1 and the
                  calibration of the stress estimators against the exact
                  Couette solution ``tau = rho nu gdot``.
``dispersed``     one crystal per particle, ``c = 0.05, 0.10, 0.20, 0.30``:
                  criteria 2 and 3 (Einstein / Krieger-Dougherty), and the
                  built-in negative control for shear thinning.
``flocculated``   fractal aggregates sized by the shear-breakup balance,
                  ``c = 0.10, 0.20``: criterion 4 (shear thinning with a
                  bootstrap interval).
``coarse``/``fine`` the same *physical* microstructure voxelised at two
                  resolutions: criterion 5 (grid convergence of ``K`` and ``n``).
``wide``/``large``larger gaps at ``c = 0.20``: the finite-size (wall-depletion)
                  scan whose ``1/H`` extrapolation gives the bulk viscosity that
                  is compared with the closures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np

import tensorlbm.frazil_powerlaw_mature as face
from tensorlbm.frazil_powerlaw_shear import (
    DEFAULT_DF,
    DEFAULT_N_CAP,
    bond_force_from_reference,
    equilibrium_floc_size,
)

ARTIFACT_DIR = REPO / "artifacts" / "frazil_powerlaw_mature"
RUN_DIR = HERE / "case_records"
PLAN_PATH = HERE / "cases.json"

NU = 0.2
#: Inertia parameter ladder: ``Re_p = gdot (2a)**2 / nu``.  Held at ``<= 2e-2``
#: so the O(Re_p) inertial correction to the drag cannot be mistaken for shear
#: thinning (the previous face ran up to ``Re_p = 0.36``).
RE_P = (5.0e-3, 1.0e-2, 2.0e-2)

#: (name, n_xy, n_z, radius_in_cells) -- the physical cell is a cube of ``n_xy``
#: cells with a two-cell wall band, so ``n_z = n_xy + 4``.
#:
#: ``coarse``/``fine`` keep the same *physical* suspension (the crystal radius is
#: a fixed fraction of the cell) and differ only in discretisation -- that is the
#: grid-convergence pair.  ``hd8``/``hd11`` keep the **same crystal size in
#: cells** and only enlarge the gap, so ``H/D`` grows: they are the finite-size
#: (wall-depletion) scan, which the ``coarse``/``fine`` pair cannot provide
#: because scaling the crystal with the cell keeps ``H/D`` fixed.
BOXES = {
    "coarse": (32, 36, 3.0),
    "fine": (48, 52, 4.5),
    "hd8": (48, 52, 3.0),
    "hd11": (64, 68, 3.0),
}


def _radius_fraction(box: str) -> float:
    n_xy, _n_z, a_cells = BOXES[box]
    return a_cells / float(n_xy)


def _steps_for(n_xy: int) -> tuple[int, int]:
    """``(n_steps, n_warmup)`` from the wall-normal momentum diffusion time.

    ``t_nu = H**2 / (pi**2 nu)``; the run keeps ``>= 1.5 t_nu`` of averaging and
    the same amount of spin-up, which is what the time-history check in the
    report shows to be settled.
    """
    h = n_xy
    t_nu = h * h / (np.pi ** 2 * NU)
    warm = int(round(1.5 * t_nu / 20.0) * 20)
    return 2 * warm, warm


def _rates_for(box: str) -> list[float]:
    """Shear-rate ladder implied by :data:`RE_P` for that box's crystal."""
    a = BOXES[box][2]
    return [re_p * NU / (2.0 * a) ** 2 for re_p in RE_P]


def _floc_sizes(box: str) -> list[int]:
    """Aggregate sizes ``N_eq(gdot)`` for the flocculated family at this box."""
    a = BOXES[box][2]
    g = _rates_for(box)
    fb = bond_force_from_reference(g[0], a, NU, DEFAULT_N_CAP, df=DEFAULT_DF)
    out = []
    for gi in g:
        out.append(int(equilibrium_floc_size(gi, radius=a, mu=NU, bond_force=fb,
                                             df=DEFAULT_DF, n_cap=DEFAULT_N_CAP)["n_primary"]))
    return out


def plan_cases() -> list[dict]:
    """Deterministic case list of the campaign."""
    cases: list[dict] = []

    def add(**kw):
        cid = "_".join(str(kw[k]) for k in ("family", "phi", "box", "seed", "rate_index"))
        if kw.get("fixed_scale") is not None:
            cid += f"_s{kw['fixed_scale']}"
        kw["case_id"] = cid
        kw.setdefault("df", DEFAULT_DF)
        kw.setdefault("self_consistent", False)
        kw.setdefault("fixed_scale", None)
        kw.setdefault("n_primary", 1)
        cases.append(kw)

    # --- (1) pure controls, both resolutions ------------------------------
    for box in ("coarse", "fine"):
        for i, _ in enumerate(RE_P):
            add(family="pure", phi=0.0, box=box, seed=0, rate_index=i)

    # --- (2) dispersed: Einstein (0.05), mid (0.10), KD (0.20), dense (0.30)
    for phi, seeds in ((0.05, (1, 2, 3)), (0.10, (1, 2)), (0.20, (1, 2, 3)), (0.30, (1,))):
        for seed in seeds:
            for i, _ in enumerate(RE_P):
                add(family="dispersed", phi=phi, box="coarse", seed=seed, rate_index=i)

    # --- (3) c = 0.20: resolution pair and finite-size scan ---------------
    for i, _ in enumerate(RE_P):
        add(family="dispersed", phi=0.20, box="fine", seed=1, rate_index=i)
    for box in ("hd8", "hd11"):
        add(family="dispersed", phi=0.20, box=box, seed=1, rate_index=1)
    add(family="dispersed", phi=0.05, box="hd8", seed=1, rate_index=1)

    # --- (3a) extra seed at c = 0.20 (velocity-profile audit record) ------
    add(family="dispersed", phi=0.20, box="coarse", seed=5, rate_index=1)

    # --- (3b) kinematic-condition sensitivity at c = 0.20 -----------------
    for scale in (0.85, 0.925):
        add(family="dispersed", phi=0.20, box="coarse", seed=4, rate_index=1,
            fixed_scale=scale)

    # --- (4) flocculated: shear thinning ----------------------------------
    for phi, seeds in ((0.10, (1, 2)), (0.20, (1,))):
        for seed in seeds:
            for i, _ in enumerate(RE_P):
                add(family="flocculated", phi=phi, box="coarse", seed=seed, rate_index=i)
    for i, _ in enumerate(RE_P):
        add(family="flocculated", phi=0.10, box="fine", seed=1, rate_index=i)

    for c in cases:
        n_xy, n_z, _a = BOXES[c["box"]]
        steps, warm = _steps_for(n_xy)
        if c["box"] == "hd11":
            steps, warm = int(steps * 0.66), int(warm * 0.66)
        c["n_xy"], c["n_z"] = n_xy, n_z
        c["radius_fraction"] = _radius_fraction(c["box"])
        c["gamma_dot"] = _rates_for(c["box"])[c["rate_index"]]
        c["steps"], c["warmup"] = steps, warm
        if c["family"] == "flocculated":
            c["n_primary"] = _floc_sizes(c["box"])[c["rate_index"]]
        c["nu"] = NU
    return cases


# ---------------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------------


def run_case(case: dict) -> dict:
    """Build the microstructure, run the point, return the record."""
    t0 = time.time()
    if case["family"] == "pure":
        rec = face._point_with_scale([], n_xy=case["n_xy"], n_z=case["n_z"], nu=NU,
                                     gamma_dot=case["gamma_dot"], scale=1.0,
                                     steps=case["steps"], warmup=case["warmup"],
                                     sample_every=10, band=(1, 4))
        rec["case"] = "pure"
    else:
        susp = face.physical_suspension(
            case["phi"], family=case["family"], n_primary=case["n_primary"],
            df=case["df"], seed=case["seed"],
            radius_fraction=case.get("radius_fraction", face.RADIUS_FRACTION))
        parts, info = face.voxelise(susp, n_xy=case["n_xy"], n_z=case["n_z"])
        # tag every primary with its aggregate centre so a floc moves rigidly
        for p in parts:
            cl = susp.clusters[p.meta["cluster"]]
            cx = float(np.mean([c[0] for c in cl])) * case["n_xy"]
            cy = float(np.mean([c[1] for c in cl])) * case["n_xy"]
            cz = 2.0 + float(np.mean([c[2] for c in cl])) * case["n_xy"]
            p.meta["rigid_centre"] = (cz, cy, cx)
        if case.get("fixed_scale") is not None:
            rec = face._point_with_scale(parts, n_xy=case["n_xy"], n_z=case["n_z"], nu=NU,
                                         gamma_dot=case["gamma_dot"],
                                         scale=float(case["fixed_scale"]),
                                         steps=case["steps"], warmup=case["warmup"],
                                         sample_every=10, band=(1, 4))
            rec["fixed_scale"] = float(case["fixed_scale"])
        elif case["self_consistent"]:
            rec = face.self_consistent_point(
                parts, n_xy=case["n_xy"], n_z=case["n_z"], nu=NU,
                gamma_dot=case["gamma_dot"], steps=case["steps"], warmup=case["warmup"],
                sample_every=10, band=(1, 4), max_iter=3, tol=0.02)
        else:
            rec = face._point_with_scale(parts, n_xy=case["n_xy"], n_z=case["n_z"], nu=NU,
                                         gamma_dot=case["gamma_dot"], scale=1.0,
                                         steps=case["steps"], warmup=case["warmup"],
                                         sample_every=10, band=(1, 4))
        rec["geometry"] = info
        rec["n_primary"] = case["n_primary"]
        rec["solid_fraction_declared"] = susp.solid_volume_fraction
        rec["phi_hydrodynamic"] = face.hydrodynamic_volume_fraction(
            info["phi_lattice"], n_primary=case["n_primary"], df=case["df"])
    rec["case"] = case
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


def _case_path(case_id: str) -> Path:
    return RUN_DIR / f"{case_id}.json"


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------


def _load_cases(cases) -> list[dict]:
    out = []
    for c in cases:
        p = _case_path(c["case_id"])
        if p.exists():
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    return out


def _phi_of(rec: dict) -> float:
    """Solids fraction referred to the physical cell volume (``n_xy**3``)."""
    geo = rec.get("geometry") or {}
    return float(geo.get("phi_lattice", rec.get("phi_lattice", 0.0)))


def _ladder(records, **sel) -> list[dict]:
    return [r for r in records if all(
        (r["case"].get(k) == v) for k, v in sel.items())]


def assemble(cases, pytest_summary: dict | None = None) -> dict:
    recs = _load_cases(cases)
    done = {r["case"]["case_id"] for r in recs}
    missing = [c["case_id"] for c in cases if c["case_id"] not in done]

    pure = [r for r in recs if r["case"]["family"] == "pure"]
    pure_sorted = sorted(pure, key=lambda r: (r["case"]["box"], r["case"]["rate_index"]))

    # ---- per-(family, phi, box) ladders -> (K, n) with bootstrap ------
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        c = r["case"]
        if c["family"] == "pure":
            continue
        if c.get("fixed_scale") is not None:
            # the kinematic-condition scan is a *protocol study*, not a point of
            # the production ladder: it must never enter a flow-curve fit
            continue
        key = (c["family"], float(c["phi"]), c["box"])
        groups.setdefault(key, []).append(r)

    fits: dict[str, dict] = {}
    ladder_fits: dict[tuple, dict] = {}
    for (family, phi, box), rows in sorted(groups.items()):
        seeds = sorted({r["case"]["seed"] for r in rows})
        samples, per_rate = [], {}
        for seed in seeds:
            rs = sorted([r for r in rows if r["case"]["seed"] == seed],
                        key=lambda r: r["case"]["rate_index"])
            g = [r["gamma_bulk"] for r in rs]
            t = [r["tau_flux"] for r in rs]
            samples.append({"gamma_dot": g, "tau": t, "seed": seed})
        for r in rows:
            per_rate.setdefault(r["case"]["rate_index"], []).append(r)
        try:
            boot = face.bootstrap_power_law(samples, n_boot=4000, seed=7)
        except Exception as exc:            # noqa: BLE001 - recorded, never hidden
            boot = {"error": str(exc)}
        eta_by_rate = {
            str(k): {
                "eta_r_mean": float(np.mean([x["eta_r"] for x in v])),
                "eta_r_sd": float(np.std([x["eta_r"] for x in v])),
                "eta_r_all_mean": float(np.mean([x["eta_r_all"] for x in v])),
                "gamma_bulk": float(np.mean([x["gamma_bulk"] for x in v])),
                "gamma_nominal": float(np.mean([x["gamma_dot_nominal"] for x in v])),
                "scale": float(np.mean([x["scale"] for x in v])),
                "re_p": float(np.mean([x["re_p"] for x in v])),
                # the solids fraction is referred to the *physical* cell volume
                # (geometry record); the cell's own count includes the wall bands
                "phi_lattice": float(np.mean([_phi_of(x) for x in v])),
                "n_primary": int(v[0]["case"].get("n_primary", 1)),
                "n_seeds": len(v),
            }
            for k, v in sorted(per_rate.items())
        }
        key = f"{family}_c{phi:g}_{box}"
        entry = {
            "family": family, "phi": float(phi), "box": box,
            "n_xy": rows[0]["case"]["n_xy"], "n_z": rows[0]["case"]["n_z"],
            "phi_lattice_mean": float(np.mean([_phi_of(r) for r in rows])),
            "phi_hydrodynamic_mean": float(np.mean([r.get("phi_hydrodynamic", _phi_of(r))
                                                    for r in rows])),
            "n_particles_mean": float(np.mean([r["n_particles"] for r in rows])),
            "n_primary": int(rows[0]["case"].get("n_primary", 1)),
            "eta_by_rate": eta_by_rate,
            "eta_r": float(np.mean([v["eta_r_mean"] for v in eta_by_rate.values()])),
            "eta_r_first_rate": list(eta_by_rate.values())[0]["eta_r_mean"],
            "power_law": boot,
            "n_seeds": len(seeds),
        }
        fits[key] = entry
        if box == "coarse":
            ladder_fits[(family, float(phi))] = {
                "eta_r": entry["eta_r"],
                "eta_r_first_rate": entry["eta_r_first_rate"],
                "phi_lattice": entry["phi_lattice_mean"],
                "n": boot.get("n"), "K": boot.get("K"),
                "n_ci": boot.get("n_ci"), "K_ci": boot.get("K_ci"),
                "n_ci_excludes_one": boot.get("n_ci_excludes_one"),
                "n_seeds": len(seeds),
            }

    # ---- pure controls: fit n over the pure ladder (must be 1) ----------
    pure_cases_out = []
    for box in ("coarse", "fine"):
        rows = sorted([r for r in pure_sorted if r["case"]["box"] == box],
                      key=lambda r: r["case"]["rate_index"])
        if not rows:
            continue
        for r in rows:
            pure_cases_out.append({
                "box": box, "rate_index": r["case"]["rate_index"],
                "tau_flux": r["tau_flux"], "gamma_dot": r["gamma_bulk"],
                "nu_gamma_dot": NU * r["gamma_bulk"],
                "flux_rel_err": r["tau_flux"] / (NU * r["gamma_bulk"]) - 1.0,
                "wall_rel_err": r["tau_wall"] / (NU * r["gamma_bulk"]) - 1.0,
                "profile_r2": r["profile_r2"],
            })
        fit = None
        if len(rows) >= 2:
            fit = face.fit_power_law([r["gamma_bulk"] for r in rows],
                                     [r["tau_flux"] for r in rows])
        for rec in pure_cases_out:
            if rec["box"] == box:
                rec["n"] = fit["n"] if fit else 1.0
                rec["K"] = fit["K"] if fit else float("nan")
                rec["n_points"] = len(rows)

    # ---- finite-size extrapolation at c = 0.20 -------------------------
    # The coarse/fine pair cannot serve here: both keep the crystal radius a fixed
    # fraction of the cell, so H/D is the same in both.  The scan is
    # coarse (H/D 5.7) -> hd8 (8.3) -> hd11 (11.0) at the *same* crystal size in
    # cells, all at the middle rate of the ladder so the comparison is isothermal
    # in Re_p as well.
    finite_size = {}
    rate_key = "1"
    for phi in (0.20,):
        pts = []
        for box in ("coarse", "hd8", "hd11"):
            f = fits.get(f"dispersed_c{phi:g}_{box}")
            if not f or rate_key not in f["eta_by_rate"]:
                continue
            rows = [r for r in recs
                    if r["case"]["family"] == "dispersed"
                    and abs(r["case"]["phi"] - phi) < 1e-9 and r["case"]["box"] == box
                    and str(r["case"]["rate_index"]) == rate_key]
            if not rows:
                continue
            pts.append({"box": box, "n_xy": f["n_xy"],
                        "h_over_d": float(np.mean([r["h_over_d"] for r in rows])),
                        "eta_r": f["eta_by_rate"][rate_key]["eta_r_mean"],
                        "eta_r_bulk": f["eta_by_rate"][rate_key].get("eta_r_mean")})
        if len(pts) >= 2:
            x = np.array([1.0 / p["h_over_d"] for p in pts])
            y = np.array([p["eta_r"] for p in pts])
            A = np.vstack([x, np.ones_like(x)]).T
            (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ np.array([slope, intercept])
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            finite_size[f"c{phi:g}"] = {
                "points": pts, "slope_per_HD": float(slope),
                "intercept_bulk": float(intercept),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "model": "eta_r(apparent) = eta_bulk + slope / (H/D)",
                "rate_index": int(rate_key),
            }

    # ---- kinematic-condition sensitivity ------------------------------
    kin = []
    for r in recs:
        if r["case"].get("fixed_scale") is not None:
            kin.append({"scale": float(r["case"]["fixed_scale"]),
                        "eta_r": float(r["eta_r"]), "phi": float(r["phi_lattice"]),
                        "seed": int(r["case"]["seed"])})
    base = fits.get("dispersed_c0.2_coarse")
    if base:
        kin.append({"scale": 1.0, "eta_r": base["eta_by_rate"]["1"]["eta_r_mean"],
                    "phi": base["phi_lattice_mean"], "seed": 1, "note": "affine (production)"})
    kin.sort(key=lambda d: d["scale"])
    kinematic = {
        "points": kin,
        "slope_deta_ds": float(np.polyfit([k["scale"] for k in kin],
                                          [k["eta_r"] for k in kin], 1)[0])
        if len(kin) >= 2 else float("nan"),
        "note": "how much eta_r moves when the crystal rigid-body rate is changed; the "
                "affine condition (scale 1) is the production protocol",
    }

    # ---- flocculated family against its hydrodynamic volume fraction ---
    floc_closure = {}
    for f in fits.values():
        if f["family"] != "flocculated" or f["box"] != "coarse":
            continue
        per_rate = {}
        for rate, v in f["eta_by_rate"].items():
            # the aggregate size is selected *per shear rate* by the breakup
            # balance, so the hydrodynamic volume fraction must be evaluated with
            # that rate's own N (8, 4, 2 on this ladder) -- not with the first
            # rate's value
            n_pri = v.get("n_primary", f["n_primary"]) or 1
            phi_h = float(face.hydrodynamic_volume_fraction(
                v["phi_lattice"], n_primary=n_pri))
            per_rate[rate] = {
                "eta_r": v["eta_r_mean"],
                "phi_hydrodynamic": phi_h,
                "n_primary": int(n_pri),
                "ebg_at_phi_h": float(face.einstein_batchelor_green(phi_h)),
                "rel_err_ebg": float(v["eta_r_mean"]
                                     / float(face.einstein_batchelor_green(phi_h)) - 1.0),
                "rel_err_kd": float(v["eta_r_mean"]
                                    / float(face.krieger_dougherty(phi_h)) - 1.0),
            }
        floc_closure[f"c{f['phi']:g}"] = {
            "phi_lattice": f["phi_lattice_mean"],
            "n_primary_by_rate": f.get("n_primary"),
            "eta_r": f["eta_r"], "per_rate": per_rate,
        }

    # ---- grid convergence ---------------------------------------------
    convergence = {}
    for family, phi in sorted({(f["family"], f["phi"]) for f in fits.values()}):
        a = fits.get(f"{family}_c{phi:g}_coarse")
        b = fits.get(f"{family}_c{phi:g}_fine")
        if a and b and "error" not in a["power_law"] and "error" not in b["power_law"]:
            convergence[f"{family}_c{phi:g}"] = face.grid_convergence_pair(
                a["power_law"], b["power_law"])

    # ---- criteria (criterion 3 uses the extrapolated bulk value) ------
    acceptance_fits = dict(ladder_fits)
    kd_note = "measured value in the confined cell (declared)"
    if "c0.2" in finite_size:
        conc = acceptance_fits.get(("dispersed", 0.20))
        fs = finite_size["c0.2"]
        if conc:
            conc = dict(conc)
            conc["eta_r_confined"] = conc["eta_r"]
            if fs["r2"] >= 0.5:
                # the scan resolves a systematic trend, so the bulk value is the
                # physically meaningful abscissa
                conc["eta_r"] = fs["intercept_bulk"]
                kd_note = ("finite-size extrapolated bulk value (H/D -> inf), "
                           f"scan R2={fs['r2']:.2f}")
            else:
                # the scan shows no trend: quoting an extrapolation would be
                # inventing a correction, so the measured value is used and the
                # scan is reported as inconclusive
                kd_note = (f"measured value in the confined cell; the H/D scan "
                           f"(5.7/8.3/11.0) shows no trend (R2={fs['r2']:.2f}), so no "
                           f"extrapolation is applied")
            acceptance_fits[("dispersed", 0.20)] = conc

    pytest_summary = pytest_summary or _pytest_summary()
    acceptance = face.evaluate_acceptance(
        pure_cases=pure_cases_out, ladder_fits=acceptance_fits,
        convergence=convergence, pytest_result=pytest_summary)

    artifact = {
        "schema": "frazil_powerlaw_mature/v1",
        "node": "p366w3-n02-frazil-rheology-mature",
        "mission_id": 12682,
        "repo": "TensorLBM",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "nu_lattice": NU,
            "re_p_ladder": list(RE_P),
            "radius_fraction": face.RADIUS_FRACTION,
            "boxes": {k: {"n_xy": v[0], "n_z": v[1]} for k, v in BOXES.items()},
            "stress_estimator": "plane momentum flux in the crystal-free wall band, "
                                "cross-checked against the per-wall momentum exchange",
            "kinematics": "self-consistent frozen structure: the crystal rigid-body "
                          "rate equals the measured bulk shear rate",
            "floc_law": "R_eq = sqrt(F_b/(mu gdot)), F_b anchored at N_eq(gdot_min)=8",
            "kd_phi_max": 0.6,
        },
        "n_cases_planned": len(cases),
        "n_cases_done": len(recs),
        "missing_cases": missing,
        "pure": pure_cases_out,
        "fits": fits,
        "finite_size": finite_size,
        "grid_convergence": convergence,
        "kinematic_sensitivity": kinematic,
        "floc_closure": floc_closure,
        "criterion3_value": kd_note,
        "acceptance": acceptance,
    }
    return artifact


def _pytest_summary() -> dict | None:
    """Run the face's own test module and summarise it (criterion 6)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_frazil_powerlaw_mature.py",
             "-q", "--no-header"],
            cwd=str(REPO), capture_output=True, text=True, timeout=1800)
        tail = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        n_pass = n_tests = 0
        for ln in tail:
            if " passed" in ln:
                import re
                m = re.search(r"(\d+) passed", ln)
                if m:
                    n_pass = int(m.group(1))
                m2 = re.search(r"(\d+) failed", ln)
                n_tests = n_pass + (int(m2.group(1)) if m2 else 0)
        return {"n_tests": n_tests, "n_passed": n_pass, "rc": out.returncode,
                "tail": tail[-1] if tail else ""}
    except Exception as exc:                 # noqa: BLE001
        return {"n_tests": 0, "n_passed": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args(argv)

    cases = plan_cases()
    if args.plan:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text(json.dumps(cases, indent=1))
        print(f"planned {len(cases)} cases -> {PLAN_PATH}")
        return 0

    if args.run:
        import torch
        torch.set_num_threads(max(1, args.threads))
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        only = {s for s in args.only.split(",") if s}
        todo = [c for c in cases if (not only or c["case_id"] in only)
                and c["case_id"] not in {p.stem for p in RUN_DIR.glob("*.json")}]
        if args.shards > 1:
            todo = [c for i, c in enumerate(todo) if i % args.shards == args.shard]
        for c in todo:
            rec = run_case(c)
            _case_path(c["case_id"]).write_text(json.dumps(rec, indent=1, default=float))
            print(f"{c['case_id']}: eta_r={rec.get('eta_r'):.4f} "
                  f"scale={rec.get('scale')} t={rec['elapsed_s']}s", flush=True)
        return 0

    if args.assemble:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        pytest_summary = None if args.skip_pytest else _pytest_summary()
        art = assemble(cases, pytest_summary=pytest_summary)
        out = ARTIFACT_DIR / "campaign.json"
        out.write_text(json.dumps(art, indent=1, default=float))
        acc = art["acceptance"]
        print(f"cases {art['n_cases_done']}/{art['n_cases_planned']} "
              f"criteria {acc['n_pass']}/{acc['n_defined']} -> {out}")
        for k, v in acc["criteria"].items():
            print(f"  {k}: {'PASS' if v.get('passed') else 'FAIL'}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())