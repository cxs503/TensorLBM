"""P41-N02 cavitation validation driver (sheet + cloud cavitation, airfoils).

Runs the case matrix of mission P41-N02 with
:class:`tensorlbm.cavitation_airfoil.MixtureCavitationSolver`, extracts
``L_cav/c``, ``delta_cav/c`` and the Strouhal number of the cavity-length
oscillation, compares them with the reference tables and writes

* ``artifacts/p41_n02/<case>.json``       -- per-case fields + metrics
* ``artifacts/p41_n02/summary.json``      -- criteria verdicts (machine readable)
* ``docs/figures/p41_n02_*.png``          -- figures for the report

Usage
-----
    python3 scripts/p41_n02_cavitation_airfoil.py --steps 600 --cpc 50 100

The driver is deterministic (fixed seed) and every number written to JSON is
produced by the solver, never transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cavitation_airfoil import (  # noqa: E402
    CavitationFlowConfig,
    MixtureCavitationSolver,
    cavity_metrics,
    reference_data,
)

ART = ROOT / "artifacts" / "p41_n02"
FIG = ROOT / "docs" / "figures"


def _cavity_from_pressure(p: torch.Tensor, cfg: CavitationFlowConfig) -> torch.Tensor:
    """Vapour indicator from the isobaric (pressure cut-off) criterion.

    ``alpha = 1`` where the mixture pressure is below the vapour pressure, with
    a one-cell smooth ramp so that the same sampling routine as the transported
    vapour field can be used.  This is the classical minimum-pressure / isobaric
    cavity criterion: the cavity is the region where the pressure would drop
    below ``p_v``.
    """
    pf = p.double()
    return torch.sigmoid((cfg.p_v - pf) / (0.05 * abs(cfg.p_v)))


def run_case(spec: dict) -> dict:
    """Run one validation case and return the metrics dictionary."""
    kind, alpha_deg, sigma, cpc, steps, mode = (
        spec["airfoil"],
        spec["alpha_deg"],
        spec["sigma"],
        spec["cells_per_chord"],
        spec["n_steps"],
        spec.get("mode", "cutoff"),
    )
    cfg = CavitationFlowConfig(
        airfoil=kind,
        alpha_deg=alpha_deg,
        sigma=sigma,
        cells_per_chord=cpc,
        domain_x_chords=6.0,
        domain_y_chords=3.0,
        le_x_chords=1.5,
        le_y_chords=1.5,
        n_steps=steps,
        cfl=spec.get("cfl", 0.2),
        spinup_steps=0 if mode == "cutoff" else 300,
        ramp_steps=0 if mode == "cutoff" else 200,
        c_dest=0.0 if mode == "cutoff" else 1.0,
        c_prod=0.0 if mode == "cutoff" else 1.0,
        sample_interval=spec.get("sample_interval", 5),
        transient_fraction=0.5,
        seed=0,
    )
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    t0 = time.time()
    diverged = False
    for k in range(cfg.n_steps):
        solver.step()
        if k % cfg.sample_interval == 0:
            solver.sample()
        if not math.isfinite(float(solver.u.abs().max())):
            diverged = True
            break
    wall = time.time() - t0

    if mode == "cutoff":
        alpha = _cavity_from_pressure(solver.p, cfg)
    else:
        alpha = solver.alpha
    m = cavity_metrics(alpha, cfg, threshold=cfg.cavity_threshold, geometry=solver.geometry)
    summary = solver.summary()
    return {
        "case": spec,
        "status": "diverged" if diverged else "ok",
        "wall_seconds": round(wall, 1),
        "steps_completed": solver._step,
        "l_cav": m["l_cav"],
        "l_cav_mean": summary["l_cav_mean"],
        "l_cav_min": summary["l_cav_min"],
        "l_cav_max": summary["l_cav_max"],
        "delta_cav": m["delta_cav"],
        "detachment_x": m["detachment_x"],
        "closure_x": m["closure_x"],
        "strouhal": summary["strouhal"],
        "st_cl": summary["st_cl"],
        "shedding_amplitude": summary["shedding_amplitude"],
        "cl_mean": summary["cl_mean"],
        "cd_mean": summary["cd_mean"],
        "div_residual": summary["div_residual"],
        "mass_balance": summary["mass_balance"],
        "p_min": float(solver.p.min()),
        "p_max": float(solver.p.max()),
        "alpha_max": float(solver.alpha.max()) if mode != "cutoff" else float(alpha.max()),
        "geometry": summary["geometry"],
        "time_series": summary["time_series"],
        "lcav_series": summary["lcav_series"],
        "cl_series": summary["cl_series"],
        "spectrum": summary["spectrum"],
        "config": asdict(cfg),
    }


def _dump(case: dict, tag: str) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    p = ART / f"{tag}.json"
    p.write_text(json.dumps(case, indent=2, default=str))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--cpc", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--modes", nargs="+", default=["cutoff"])
    args = ap.parse_args()

    ref = reference_data()
    specs: list[dict] = []
    for cpc in args.cpc:
        for s in ref["sheet"]["sigma"]:
            for mode in args.modes:
                specs.append(
                    dict(
                        airfoil="NACA0015",
                        alpha_deg=5.0,
                        sigma=s,
                        cells_per_chord=cpc,
                        n_steps=args.steps,
                        cfl=args.cfl,
                        mode=mode,
                    )
                )
        for s in ref["cloud"]["sigma"]:
            for mode in args.modes:
                specs.append(
                    dict(
                        airfoil="ClarkY",
                        alpha_deg=6.0,
                        sigma=s,
                        cells_per_chord=cpc,
                        n_steps=args.steps,
                        cfl=args.cfl,
                        mode=mode,
                    )
                )

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_case, sp): sp for sp in specs}
        for fut in futs:
            sp = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                res = {"case": sp, "status": f"error: {exc}"}
            tag = (
                f"{sp['airfoil']}_a{sp['alpha_deg']:g}_s{sp['sigma']:g}"
                f"_cpc{sp['cells_per_chord']}_{sp['mode']}"
            )
            _dump(res, tag)
            results.append(res)
            print(
                f"{tag:52s} {res.get('status'):>8s} "
                f"L/c={res.get('l_cav', float('nan')):.3f} "
                f"d/c={res.get('delta_cav', float('nan')):.4f} "
                f"St={res.get('strouhal', float('nan')):.3f}",
                flush=True,
            )

    # merge with any per-case JSON already on disk (e.g. a finer-mesh run in a
    # separate invocation) so the summary always covers the whole matrix
    merged = {f"{r['case']['airfoil']}_{r['case']['alpha_deg']:g}_s{r['case']['sigma']:g}"
              f"_cpc{r['case']['cells_per_chord']}_{r['case'].get('mode', 'cutoff')}": r
              for r in results}
    for pf in ART.glob("*.json"):
        if pf.name == "summary.json":
            continue
        try:
            d = json.loads(pf.read_text())
            c = d.get("case") or {}
            if "sigma" in c and "cells_per_chord" in c:
                merged.setdefault(pf.stem, d)
        except Exception:
            continue
    results = list(merged.values())
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["criteria"], indent=2))
    make_figures(results, ref)
    return 0


def _pick(results: list[dict], airfoil: str, sigma: float, cpc: int, mode: str = "cutoff"):
    for r in results:
        c = r.get("case", {})
        if (
            c.get("airfoil") == airfoil
            and abs(c.get("sigma", -1) - sigma) < 1e-9
            and c.get("cells_per_chord") == cpc
            and c.get("mode", "cutoff") == mode
        ):
            return r
    return None


def build_summary(results: list[dict], ref: dict) -> dict:
    """Evaluate the P41-N02 acceptance criteria against the reference tables."""
    sheet = ref["sheet"]
    cloud = ref["cloud"]
    cpcs = sorted({r["case"]["cells_per_chord"] for r in results if "case" in r})
    finest = cpcs[-1] if cpcs else 50
    crits: dict[str, dict] = {}

    # (a) L_cav/c vs Brennen, (b) delta_cav/c vs Brennen -- finest mesh
    rows = []
    for s, lref, dref, lmer in zip(
        sheet["sigma"], sheet["l_cav_exp"], sheet["delta_cav_exp"], sheet["l_cav_merkle"]
    ):
        r = _pick(results, "NACA0015", s, finest)
        lsim = r["l_cav"] if r and r.get("status") == "ok" else float("nan")
        dsim = r["delta_cav"] if r and r.get("status") == "ok" else float("nan")
        rows.append(
            {
                "sigma": s,
                "l_cav_sim": lsim,
                "l_cav_ref": lref,
                "l_cav_merkle": lmer,
                "err_l_cav_pct": 100.0 * abs(lsim - lref) / lref if lref else float("nan"),
                "delta_cav_sim": dsim,
                "delta_cav_ref": dref,
                "err_delta_cav_pct": 100.0 * abs(dsim - dref) / dref if dref else float("nan"),
            }
        )
    errs_l = [r["err_l_cav_pct"] for r in rows if not math.isnan(r["err_l_cav_pct"])]
    errs_d = [r["err_delta_cav_pct"] for r in rows if not math.isnan(r["err_delta_cav_pct"])]
    crits["a_l_cav_vs_brennen"] = {
        "threshold_pct": 20.0,
        "max_err_pct": max(errs_l) if errs_l else float("nan"),
        "verdict": "PASS" if errs_l and max(errs_l) < 20.0 else "FAIL",
        "rows": rows,
    }
    crits["b_delta_cav_vs_brennen"] = {
        "threshold_pct": 25.0,
        "max_err_pct": max(errs_d) if errs_d else float("nan"),
        "verdict": "PASS" if errs_d and max(errs_d) < 25.0 else "FAIL",
    }

    # (c) Strouhal vs Franc
    st_rows = []
    for s, stref in zip(cloud["sigma"], cloud["st_exp"]):
        r = _pick(results, "ClarkY", s, finest)
        st = r.get("strouhal", float("nan")) if r and r.get("status") == "ok" else float("nan")
        amp = r.get("shedding_amplitude", 0.0) if r else 0.0
        st_rows.append(
            {
                "sigma": s,
                "st_sim": st,
                "st_ref": stref,
                "err_pct": 100.0 * abs(st - stref) / stref if stref and not math.isnan(st) else float("nan"),
                "shedding_amplitude": amp,
            }
        )
    st_errs = [r["err_pct"] for r in st_rows if not math.isnan(r["err_pct"])]
    crits["c_strouhal_vs_franc"] = {
        "threshold_pct": 15.0,
        "max_err_pct": max(st_errs) if st_errs else float("nan"),
        "verdict": "PASS" if st_errs and max(st_errs) < 15.0 else "FAIL",
        "rows": st_rows,
    }

    # (d) monotonicity sigma down -> L_cav up
    mono = []
    for cp in cpcs:
        seq = []
        for s in sheet["sigma"]:
            r = _pick(results, "NACA0015", s, cp)
            seq.append(r["l_cav"] if r and r.get("status") == "ok" else float("nan"))
        ok = all(
            seq[i] < seq[i + 1] for i in range(len(seq) - 1) if not math.isnan(seq[i + 1])
        )
        mono.append({"cells_per_chord": cp, "l_cav_by_sigma": seq, "monotone": ok})
    crits["d_monotonic_l_cav"] = {
        "verdict": "PASS" if mono and all(m["monotone"] for m in mono) else "FAIL",
        "rows": mono,
    }

    # (e) mesh convergence of L_cav
    conv = []
    for s in sheet["sigma"]:
        vals = []
        for cp in cpcs:
            r = _pick(results, "NACA0015", s, cp)
            vals.append(r["l_cav"] if r and r.get("status") == "ok" else float("nan"))
        finite = [v for v in vals if not math.isnan(v)]
        rel = (max(finite) - min(finite)) / max(finite) if len(finite) > 1 and max(finite) > 0 else float("nan")
        conv.append({"sigma": s, "l_cav_by_cpc": vals, "rel_change": rel})
    rels = [c["rel_change"] for c in conv if not math.isnan(c["rel_change"])]
    crits["e_mesh_convergence"] = {
        "threshold_pct": 15.0,
        "max_rel_change_pct": 100.0 * max(rels) if rels else float("nan"),
        "verdict": "PASS" if rels and max(rels) < 0.15 else "FAIL",
        "rows": conv,
    }

    # (f) mass conservation
    mb = [r.get("mass_balance", float("nan")) for r in results if r.get("status") == "ok"]
    dv = [r.get("div_residual", float("nan")) for r in results if r.get("status") == "ok"]
    worst = max([v for v in mb + dv if v == v]) if (mb or dv) else float("nan")
    crits["f_mass_conservation"] = {
        "threshold": 1.0e-4,
        "max_residual": worst,
        "mass_balance_max": max(mb) if mb else float("nan"),
        "divergence_residual_max": max(dv) if dv else float("nan"),
        "verdict": "PASS" if worst == worst and worst < 1e-4 else "FAIL",
    }

    return {
        "mission": "P41-N02",
        "reference": ref,
        "mesh_cells_per_chord": cpcs,
        "finest_mesh": finest,
        "n_cases": len(results),
        "cases_ok": sum(1 for r in results if r.get("status") == "ok"),
        "criteria": crits,
    }


def make_figures(results: list[dict], ref: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    sheet = ref["sheet"]
    cloud = ref["cloud"]
    cpcs = sorted({r["case"]["cells_per_chord"] for r in results if "case" in r})
    finest = cpcs[-1]

    # fig 1: L_cav vs sigma (sim vs Brennen vs Merkle)
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(sheet["sigma"], sheet["l_cav_exp"], "ks-", label="Brennen 1994 (exp)")
    ax.plot(sheet["sigma"], sheet["l_cav_merkle"], "b^--", label="Merkle 1998 (num)")
    for cp, mk in zip(cpcs, ["o", "s", "d"]):
        sig, l = [], []
        for s in sheet["sigma"]:
            r = _pick(results, "NACA0015", s, cp)
            if r and r.get("status") == "ok":
                sig.append(s)
                l.append(r["l_cav"])
        ax.plot(sig, l, mk + "-", label=f"this work, dx=c/{cp}", alpha=0.85)
    ax.set_xlabel(r"cavitation number $\sigma$")
    ax.set_ylabel(r"$L_{cav}/c$")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "p41_n02_lcav_vs_sigma.png", dpi=160)
    plt.close(fig)

    # fig 2: delta_cav vs sigma
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(sheet["sigma"], sheet["delta_cav_exp"], "ks-", label="Brennen 1994 (exp)")
    r = [_pick(results, "NACA0015", s, finest) for s in sheet["sigma"]]
    ax.plot(
        sheet["sigma"], [x["delta_cav"] if x else float("nan") for x in r], "o-", label=f"dx=c/{finest}"
    )
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\delta_{cav}/c$")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p41_n02_delta_vs_sigma.png", dpi=160)
    plt.close(fig)

    # fig 3: time history of the cavity length (cloud cavitation case)
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for s in cloud["sigma"]:
        rr = _pick(results, "ClarkY", s, finest)
        if rr and rr.get("time_series"):
            ax.plot(rr["time_series"], rr["lcav_series"], label=fr"Clark-Y $\sigma$={s}")
    ax.set_xlabel("t U/c")
    ax.set_ylabel(r"$L_{cav}/c$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p41_n02_cloud_history.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())