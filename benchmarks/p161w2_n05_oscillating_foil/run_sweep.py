#!/usr/bin/env python
"""Campaign driver for node ``p161w2-arb-n05`` (oscillating-foil propulsion).

Runs every LBM case of the verification face (one OS process per case; the
numpy solver is single threaded so there is no oversubscription), evaluates the
machine-checkable criteria C1..C8 from
:mod:`tensorlbm.oscillating_foil_criteria`, and writes

* ``artifacts/<label>.json``   one payload per case (numbers only, no prose)
* ``artifacts/analytic.json``  closed-form anchor tables
* ``artifacts/results.json``   criteria verdicts + the numbers behind them

Usage::

    python run_sweep.py --workers 12                # full matrix
    python run_sweep.py --only heave_h025_k1 --cycles 2   # quick single case
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
sys.path.insert(0, SRC)

from tensorlbm.flapping_propulsion import FlappingKinematics, garrick_coefficients  # noqa: E402
from tensorlbm.oscillating_foil_criteria import (  # noqa: E402
    criterion_c1_linear_region,
    criterion_c5b_thrust_reversal,
    criterion_c9_momentum_balance,
    criterion_c1b_amplitude_scaling,
    criterion_c2_efficiency_peak,
    criterion_c3_degenerate_limits,
    criterion_c4_energy_ledger,
    criterion_c5_strouhal_monotone,
    criterion_c6_grid_convergence,
    criterion_c7_negative_controls,
    criterion_c8_reproducibility,
    summarise,
)
from tensorlbm.oscillating_foil_lbm import (  # noqa: E402
    OscillatingFoil2D,
    OscillatingFoilConfig,
    cycle_statistics,
    dissipation_neq_rate,
)
from tensorlbm.flapping_lbm_2d import momenta  # noqa: E402

# ---------------------------------------------------------------------------
# physics set-up shared by every case
# ---------------------------------------------------------------------------
U_INF = 0.15
RE = 500.0
PITCH_AXIS_A = -0.5

#: base case list.  ``dx``/``dy`` are the domain extents in chords.
CASES: list[dict] = [
    # --- reference / controls ------------------------------------------------
    dict(label="lock_ref", mode="locked", k=1.0, cycles=6),
    dict(label="lock_ref_c24", mode="locked", k=1.0, chord=24, cycles=6),
    dict(label="lock_ref_c42", mode="locked", k=1.0, chord=42, cycles=6),
    dict(label="zero_motion", mode="locked", k=1.0, h0c=0.0, a0=0.0, cycles=3),
    dict(label="nc_noibm", mode="heave", k=1.0, h0c=0.15, cycles=3, noibm=True),
    dict(label="nc_heave_flip", mode="heave", k=1.0, h0c=0.15, cycles=8, phase_h=180.0),
    # --- small-amplitude linear region (C1) ---------------------------------
    dict(label="lin_heave_02", mode="heave", k=1.0, h0c=0.02, cycles=8),
    dict(label="lin_heave_04", mode="heave", k=1.0, h0c=0.04, cycles=8),
    dict(label="lin_pitch_2deg", mode="pitch", k=1.0, a0=2.0, cycles=8),
    dict(label="lin_coupled_02", mode="coupled", k=1.0, h0c=0.02, a0=2.0, cycles=8),
    # --- reduced frequency / Strouhal sweep at fixed heave amplitude --------
    dict(label="heave_h015_k0.75", mode="heave", k=0.75, h0c=0.15, cycles=8),
    dict(label="heave_h015_k1", mode="heave", k=1.0, h0c=0.15, cycles=8),
    dict(label="heave_h015_k1.5", mode="heave", k=1.5, h0c=0.15, cycles=8),
    dict(label="heave_h015_k2", mode="heave", k=2.0, h0c=0.15, cycles=8),
    dict(label="heave_h015_k3", mode="heave", k=3.0, h0c=0.15, cycles=8),
    # --- amplitude (St) sweep at fixed k ------------------------------------
    dict(label="heave_h005_k1", mode="heave", k=1.0, h0c=0.05, cycles=8),
    dict(label="heave_h010_k1", mode="heave", k=1.0, h0c=0.10, cycles=8),
    dict(label="heave_h020_k1", mode="heave", k=1.0, h0c=0.20, cycles=8),
    dict(label="heave_h025_k1", mode="heave", k=1.0, h0c=0.25, cycles=8),
    # --- thrust reversal of a purely pitching section -----------------------
    dict(label="pitch_a10_k0.75", mode="pitch", k=0.75, a0=10.0, cycles=8),
    dict(label="pitch_a10_k1", mode="pitch", k=1.0, a0=10.0, cycles=8),
    dict(label="pitch_a10_k2", mode="pitch", k=2.0, a0=10.0, cycles=8),
    # --- coupled pitch-plunge, both phases ----------------------------------
    dict(label="cpl_phi90_k1", mode="coupled", k=1.0, h0c=0.15, a0=10.0, phi=90.0, cycles=8),
    dict(label="cpl_phi270_k1", mode="coupled", k=1.0, h0c=0.15, a0=10.0, phi=270.0, cycles=8),
    # --- grid convergence (C6): same physics on three chords ----------------
    dict(label="gci_c24", mode="heave", k=1.0, h0c=0.15, chord=24, cycles=8),
    dict(label="gci_c42", mode="heave", k=1.0, h0c=0.15, chord=42, cycles=8),
    # --- energy ledger in a closed periodic box (C4) ------------------------
    dict(label="ebox_c30", mode="heave", k=1.0, h0c=0.10, chord=30, cycles=4, warmup=1,
         periodic=True, dx=3.0, dy=2.0),
    dict(label="ebox_c45", mode="heave", k=1.0, h0c=0.10, chord=45, cycles=4, warmup=1,
         periodic=True, dx=3.0, dy=2.0),
    # --- bit-level reproducibility (C8) -------------------------------------
    dict(label="rep_a", mode="heave", k=1.0, h0c=0.15, cycles=8),
]


def case_cfg(spec: dict, default_chord: int) -> OscillatingFoilConfig:
    return OscillatingFoilConfig(
        chord=spec.get("chord", default_chord),
        re=RE,
        U=U_INF,
        h0_over_c=spec.get("h0c", 0.25),
        alpha0_deg=spec.get("a0", 10.0),
        k=spec.get("k", 1.0),
        phi_deg=spec.get("phi", 90.0),
        phi_h_deg=spec.get("phase_h", 0.0),
        mode=spec["mode"],
        n_cycles=spec.get("cycles", 8),
        warmup_cycles=spec.get("warmup", 2),
        domain_x_over_c=spec.get("dx", 9.0),
        domain_y_over_c=spec.get("dy", 4.0),
        periodic_x=bool(spec.get("periodic", False)),
        periodic_y=bool(spec.get("periodic", False)),
        track_neq=bool(spec.get("periodic", False)) or spec.get("track_neq", False),
        solid_interior=spec.get("solid", True),
        kernel=spec.get("kernel", "4pt"),
        n_forcing_passes=spec.get("passes", 1),
    )


def theory_for(cfg: OscillatingFoilConfig) -> dict:
    if cfg.mode == "locked":
        return {"C_T": 0.0, "C_P": 0.0, "eta": float("nan")}
    kin = FlappingKinematics(
        U=cfg.U,
        c=float(cfg.chord),
        h0=cfg.motion.h0,
        alpha0=cfg.motion.alpha0,
        phi=cfg.motion.phi,
        a=cfg.motion.a,
    )
    g = garrick_coefficients(kin, cfg.k)
    return {
        "C_T": g.C_T,
        "C_S": g.C_S,
        "C_T_tilt": g.C_T_tilt,
        "C_P": g.C_P,
        "eta": g.eta,
        "F": g.C.real,
        "G": g.C.imag,
        "absC": abs(g.C),
        "strouhal": kin.strouhal(cfg.k),
    }


def run_case(spec: dict, default_chord: int, diag_stride: int = 3) -> dict:
    cfg = case_cfg(spec, default_chord)
    solver = OscillatingFoil2D(cfg)
    if spec.get("noibm"):
        solver._bx = solver._by = np.zeros(0)
        solver.cfg.solid_interior = False
    per = cfg.steps_per_cycle()
    rows = []
    t0 = time.time()
    for _ in range(cfg.total_steps()):
        rows.append(solver.step(diagnostics=True))
    wall = time.time() - t0

    stats = cycle_statistics(rows, cfg)
    theory = theory_for(cfg)

    # energy ledger over the whole averaging window (integer cycles)
    use = rows
    n_use = (len(use) // per) * per
    win = use[-n_use:]
    work_fluid = float(np.sum([r["work_fluid"] for r in win]))
    work_mech = float(np.sum([r["body_work"] for r in win]))
    phi_cont = float(np.sum([r["phi"] for r in win]))
    phi_neq = float(np.sum([r.get("phi_neq", 0.0) for r in win]))
    flux = float(np.sum([r["flux"] for r in win]))
    dke = float(win[-1]["ke"] - win[0]["ke"])

    extra = {}
    # independent control-volume estimate of the mean streamwise force over the
    # last full cycle (same number as the IBM force if momentum is conserved)
    cv = [r["cv_in"] - r["cv_out"] for r in win[-per:]]
    q_ref = 0.5 * cfg.U**2 * cfg.chord
    # control-volume estimate of the *drag* coefficient (positive = drag);
    # C_T = -C_D for the same flow
    extra["C_D_control_volume"] = float(np.mean(cv) / q_ref)
    extra["C_T_control_volume"] = -extra["C_D_control_volume"]
    if cfg.mode == "locked" and not spec.get("noibm"):
        rho, ux, uy = momenta(solver.f)
        p = rho / 3.0
        x1, x2 = 2, solver.nx - 3
        mom1 = float(np.sum(rho[:, x1] * ux[:, x1] ** 2 + p[:, x1]))
        mom2 = float(np.sum(rho[:, x2] * ux[:, x2] ** 2 + p[:, x2]))
        q = 0.5 * cfg.U**2 * cfg.chord
        extra["C_D_control_volume"] = (mom1 - mom2) / q
        rho, ux, uy = momenta(solver.f)
        extra["u_max"] = float(np.max(np.hypot(ux, uy)))

    stride = max(1, per // 64)
    series = [
        {"s": r["step"], "fx": r["Fx_fluid"], "fy": r["Fy_fluid"], "w": r["body_work"]}
        for r in rows[::stride]
    ]
    payload = {
        "label": spec["label"],
        "node": "p161w2-arb-n05",
        "config": {
            "chord": cfg.chord,
            "re": cfg.re,
            "U": cfg.U,
            "k": cfg.k,
            "h0_over_c": cfg.h0_over_c,
            "alpha0_deg": cfg.alpha0_deg,
            "phi_deg": cfg.phi_deg,
            "phi_h_deg": cfg.phi_h_deg,
            "mode": cfg.mode,
            "periodic": bool(cfg.periodic_x and cfg.periodic_y),
            "noibm": bool(spec.get("noibm")),
            "n_cycles": cfg.n_cycles,
            "warmup_cycles": cfg.warmup_cycles,
            "steps_per_cycle": per,
        },
        "grid": [solver.nx, solver.ny],
        "tau": cfg.tau,
        "nu": cfg.nu,
        "steps": cfg.total_steps(),
        "wall_s": wall,
        "ms_per_step": 1000.0 * wall / max(1, cfg.total_steps()),
        "coeffs": stats,
        "theory": theory,
        "energy": {
            "work_fluid": work_fluid,
            "work_mech": work_mech,
            "dke": dke,
            "phi_cont": phi_cont,
            "phi_neq": phi_neq,
            "flux": flux,
            "cycles": n_use / per,
        },
        "extra": extra,
        "series": series,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


#: fields that are wall-clock bookkeeping, not physics -> excluded from the
#: reproducibility hash (criterion C8)
_TIMING_KEYS = (
    "wall_s",
    "ms_per_step",
    "sha256",
    "label",
    "node",
    "python",
    "numpy",
)


def canonical_sha256(payload: dict) -> str:
    import hashlib as _h

    body = {k: v for k, v in payload.items() if k not in _TIMING_KEYS}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return _h.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# criteria evaluation
# ---------------------------------------------------------------------------
def evaluate(payloads: dict[str, dict], analytic: dict, outdir: str) -> dict:
    def need(label):
        return payloads[label]

    def coeffs(label):
        return need(label)["coeffs"]

    cd0 = -coeffs("lock_ref")["C_T"]

    def lin_case(label):
        p = need(label)
        c = p["coeffs"]
        return {
            "label": label,
            "C_T": c["C_T"],
            "cd0": -coeffs("lock_ref")["C_T"],
            "theory": p["theory"],
            "C_T_cycle_std": c["C_T_cycle_std"],
            "h0_over_c": p["config"]["h0_over_c"],
            "k": p["config"]["k"],
            "strouhal": c["strouhal"],
        }

    # --- C1 / C1b: small-amplitude linear region ----------------------------
    c1 = criterion_c1_linear_region(
        [lin_case(lbl) for lbl in ("lin_heave_02", "lin_pitch_2deg", "lin_coupled_02")]
    )
    c1b = criterion_c1b_amplitude_scaling([lin_case("lin_heave_02"), lin_case("lin_heave_04")])

    # --- C2: efficiency vs reduced frequency --------------------------------
    eta_labels = [
        "heave_h015_k0.75",
        "heave_h015_k1",
        "heave_h015_k1.5",
        "heave_h015_k2",
        "heave_h015_k3",
    ]
    c2 = criterion_c2_efficiency_peak(
        [
            {"k": need(lbl)["config"]["k"], "eta": coeffs(lbl)["eta"], "theory": need(lbl)["theory"]}
            for lbl in eta_labels
        ]
    )

    # --- C3: degenerate limits ---------------------------------------------
    # C3 is a property of the analytic anchor: the LBM "zero motion" case still
    # carries the steady drag of a body in a stream (that is C7a's job), so the
    # vanishing-force limit is checked by the body-free control in C7b instead.
    c3 = criterion_c3_degenerate_limits(
        {
            "quasi_steady_rel_error": analytic["summary"]["quasi_steady_rel_error"],
            "C_at_zero_minus_one": analytic["summary"]["C_at_zero_minus_one"],
            "C_at_inf_minus_half": analytic["summary"]["C_at_inf_minus_half"],
            "pure_heave_rel_error": analytic["summary"]["pure_heave_rel_error"],
        },
        None,
    )

    # --- C4: energy ledger in a closed box, two resolutions -----------------
    ledger_rows = []
    for lbl in ("ebox_c30", "ebox_c45"):
        e = need(lbl)["energy"]
        row = criterion_c4_energy_ledger(
            {
                "work": e["work_fluid"],
                "dke": e["dke"],
                "phi": e["phi_neq"] if abs(e["phi_neq"]) > 0 else e["phi_cont"],
                "flux": e["flux"],
                "cycles": e["cycles"],
            }
        )
        row["label"] = lbl
        ledger_rows.append(row)
    c4 = dict(ledger_rows[-1])
    shrinks = (
        ledger_rows[-1]["value"]["rel_residual"] <= ledger_rows[0]["value"]["rel_residual"]
    )
    c4["value"]["resolutions"] = [
        {"label": r["label"], "rel_residual": r["value"]["rel_residual"], "verdict": r["verdict"]}
        for r in ledger_rows
    ]
    c4["value"]["residual_decreases_with_resolution"] = shrinks
    c4["verdict"] = "PASS" if (ledger_rows[-1]["verdict"] == "PASS" and shrinks) else "FAIL"

    # --- C5: Strouhal monotonicity (amplitude sweep at fixed k) -------------
    st_labels = ["heave_h005_k1", "heave_h010_k1", "heave_h015_k1", "heave_h020_k1",
                 "heave_h025_k1"]
    c5 = criterion_c5_strouhal_monotone(
        [
            {
                "label": lbl,
                "strouhal": coeffs(lbl)["strouhal"],
                "C_T": coeffs(lbl)["C_T"],
                "C_T_cycle_std": coeffs(lbl)["C_T_cycle_std"],
            }
            for lbl in st_labels
        ]
    )
    c5b = criterion_c5b_thrust_reversal(
        [
            {
                "k": need(lbl)["config"]["k"],
                "C_T": coeffs(lbl)["C_T"],
                "theory": need(lbl)["theory"],
            }
            for lbl in ("pitch_a10_k0.75", "pitch_a10_k1", "pitch_a10_k2")
        ]
    )

    # --- C6: grid convergence of the thrust increment -----------------------
    def dct(label, lock_label):
        return coeffs(label)["C_T"] - coeffs(lock_label)["C_T"]

    c6 = criterion_c6_grid_convergence(
        [
            {"chord": need("gci_c24")["config"]["chord"], "value": dct("gci_c24", "lock_ref_c24")},
            {"chord": need("heave_h015_k1")["config"]["chord"],
             "value": dct("heave_h015_k1", "lock_ref")},
            {"chord": need("gci_c42")["config"]["chord"], "value": dct("gci_c42", "lock_ref_c42")},
        ]
    )
    c6["value"]["quantity"] = "thrust increment dC_T = C_T + C_D0 (per-grid locked reference)"

    # --- C7: negative-control trio ------------------------------------------
    c7 = criterion_c7_negative_controls(
        {
            "C_T": coeffs("lock_ref")["C_T"],
            "cd_independent": need("lock_ref")["extra"].get("C_D_control_volume"),
        },
        {"C_T": coeffs("nc_noibm")["C_T"], "C_P": coeffs("nc_noibm")["C_P"]},
        {"C_T": coeffs("nc_heave_flip")["C_T"]},
        {"C_T": coeffs("heave_h015_k1")["C_T"]},
    )

    # --- C8: bit-level reproducibility --------------------------------------
    c8 = criterion_c8_reproducibility(
        {"sha256": need("heave_h015_k1")["sha256"], "C_T": coeffs("heave_h015_k1")["C_T"]},
        {"sha256": need("rep_a")["sha256"], "C_T": coeffs("rep_a")["C_T"]},
    )

    # --- C9: IBM force vs control-volume momentum balance -------------------
    mb_labels = [
        lbl
        for lbl in payloads
        if payloads[lbl]["extra"].get("C_T_control_volume") is not None
        and not payloads[lbl]["config"]["periodic"]
    ]
    c9 = criterion_c9_momentum_balance(
        [
            {
                "label": lbl,
                "C_T": coeffs(lbl)["C_T"],
                "C_T_control_volume": need(lbl)["extra"].get(
                    "C_T_control_volume",
                    -need(lbl)["extra"].get("C_D_control_volume", 0.0),
                ),
            }
            for lbl in mb_labels
        ]
    )

    crit = [c1, c1b, c2, c3, c4, c5, c5b, c6, c7, c8, c9]
    return {
        "node": "p161w2-arb-n05",
        "set_up": {
            "U": U_INF,
            "Re": RE,
            "pitch_axis_a": PITCH_AXIS_A,
            "default_chord": need("heave_h015_k1")["config"]["chord"],
            "grid": need("heave_h015_k1")["grid"],
            "steps_per_cycle": need("heave_h015_k1")["config"]["steps_per_cycle"],
            "cycles_averaged": coeffs("heave_h015_k1")["n_cycles_averaged"],
            "cd0_locked": cd0,
        },
        "cases": {
            label: {
                "grid": p["grid"],
                "mode": p["config"]["mode"],
                "k": p["config"]["k"],
                "h0_over_c": p["config"]["h0_over_c"],
                "alpha0_deg": p["config"]["alpha0_deg"],
                "phi_deg": p["config"]["phi_deg"],
                "C_T": p["coeffs"]["C_T"],
                "C_L": p["coeffs"]["C_L"],
                "C_P": p["coeffs"]["C_P"],
                "eta": p["coeffs"]["eta"],
                "C_T_theory": p["theory"]["C_T"],
                "C_P_theory": p["theory"]["C_P"],
                "eta_theory": p["theory"]["eta"],
                "strouhal": p["coeffs"]["strouhal"],
                "C_T_cycle_std": p["coeffs"]["C_T_cycle_std"],
                "C_T_control_volume": p["extra"].get("C_T_control_volume"),
                "sha256": p["sha256"],
            }
            for label, p in payloads.items()
        },
        "criteria": crit,
        "summary": summarise(crit),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--chord", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="")
    ap.add_argument("--cycles", type=int, default=0, help="override cycle count")
    ap.add_argument("--diag-stride", type=int, default=1)
    ap.add_argument("--skip-criteria", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    specs = CASES
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        specs = [s for s in CASES if s["label"] in keep]
        if not specs:
            raise SystemExit(f"no case matches {args.only!r}")
    if args.cycles:
        specs = [{**s, "cycles": args.cycles} for s in specs]

    analytic_path = os.path.join(args.outdir, "analytic.json")
    if not os.path.exists(analytic_path):
        sys.path.insert(0, os.path.join(HERE, "tools"))
        from analytic_anchor import build  # noqa: PLC0415

        with open(analytic_path, "w") as fh:
            json.dump(build(), fh, indent=1)
    with open(analytic_path) as fh:
        analytic = json.load(fh)

    t0 = time.time()
    payloads: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_case, spec, args.chord, args.diag_stride): spec["label"] for spec in specs
        }
        for fut in as_completed(futs):
            label = futs[fut]
            payload = fut.result()
            payloads[label] = payload
            with open(os.path.join(args.outdir, f"{label}.json"), "w") as fh:
                json.dump(payload, fh)
            c = payload["coeffs"]
            print(
                f"[{time.time() - t0:7.1f}s] {label:18s} grid={payload['grid']} "
                f"C_T={c['C_T']:+.4f} C_P={c['C_P']:+.4f} eta={c['eta']:+.3f} "
                f"th={payload['theory']['C_T']:+.4f} {payload['ms_per_step']:.1f}ms/step",
                flush=True,
            )

    if args.skip_criteria:
        return 0
    missing = {s["label"] for s in CASES} - set(payloads)
    if missing:
        print(f"criteria skipped, missing cases: {sorted(missing)}", flush=True)
        return 0
    results = evaluate(payloads, analytic, args.outdir)
    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results["summary"], indent=1), flush=True)
    for crit in results["criteria"]:
        print(f"  {crit['id']:4s} {crit['verdict']:9s} {crit['name']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())