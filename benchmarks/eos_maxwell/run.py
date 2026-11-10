#!/usr/bin/env python
"""EOS-Maxwell construction stability probe — campaign driver.

Modes
-----
``--mode pilot``     fast smoke: one tier, short hold, small box (CI/pytest scale)
``--mode tiers``     the three target-density-ratio tiers (10/100/1000) x forcing
``--mode boundary``  stability boundary map: T_r sweep x forcing, collapse onset
``--mode controls``  planar-interface control runs (lattice-selected coexistence)
``--mode all``       controls + tiers + boundary (production)

Every artefact is written incrementally to ``<out>/results.json`` so a killed
campaign still leaves auditable evidence.  Numbers are never transcribed by
hand: the file is produced by this script, and ``plot_figures.py`` consumes it.

Usage::

    python run.py --mode pilot --out out --device cpu
    python run.py --mode all   --out out --device cpu --steps 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import probe_lib as P  # noqa: E402

ALL_MODES = (
    "pilot", "tiers", "boundary", "controls", "widths", "longhold", "schemes",
    "repeats", "threads", "robust", "tau", "all",
)


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def _mk_spec(tag, mw, args, *, forcing, geometry="droplet", steps=None, L=None, R=None) -> P.RunSpec:
    return P.RunSpec(
        tag=tag,
        eos=args.eos,
        target_ratio=mw.get("target_ratio"),
        T_r=mw.get("T_r"),
        L=L if L is not None else args.L,
        R=R if R is not None else args.R,
        forcing=forcing,
        steps=steps if steps is not None else args.steps,
        sample=args.sample,
        geometry=geometry,
        device=args.device,
        width=getattr(args, "width", 4.0),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="all",
        help=(
            "comma separated subset of "
            "pilot,tiers,boundary,controls,widths,longhold,schemes,repeats,threads,"
            "robust,tau,all (a split campaign may be run as two cluster jobs)"
        ),
    )
    ap.add_argument("--out", default=str(_HERE / "out"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eos", default="pr")
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--forcings", default="edm,guo")
    ap.add_argument("--tau-list", default="0.6,0.8,1.0,1.6,2.0")
    ap.add_argument("--tau-forcings", default="edm")
    ap.add_argument("--tr-sweep", default="0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.42,0.40")
    ap.add_argument("--boundary-steps", type=int, default=4000)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--widths", default="4,6,8", help="initial interface widths for the construction-stability arm")
    ap.add_argument("--width", type=float, default=4.0, help="interface width of the tier runs")
    ap.add_argument("--control-width", type=float, default=8.0)
    ap.add_argument("--control-steps", type=int, default=40000)
    ap.add_argument("--schemes", default="pressure_tensor", help="extra force-discretisation schemes to probe")
    ap.add_argument("--longhold-steps", type=int, default=80000)
    ap.add_argument("--longhold-widths", default="4,6")
    ap.add_argument("--repeat-widths", default="4,6")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--thread-list", default="1,2,4,8")
    ap.add_argument("--noise-list", default="1e-7,1e-5")
    ap.add_argument("--noise-seeds", default="0,1")
    ap.add_argument("--resume", action="store_true", help="reuse results already present in results.json")
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "results.json"
    result: dict = {}
    if args.resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    result.setdefault("benchmark", "eos_maxwell_droplet_stability_probe")
    result.setdefault("eos", args.eos)
    result["prereg"] = P.PREREG
    result["environment"] = P.environment()
    result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["generated_utc"] = datetime.now(timezone.utc).isoformat()
    result.setdefault("maxwell_tiers", {})
    result.setdefault("tiers", [])
    result.setdefault("controls", [])
    result.setdefault("boundary", [])
    result.setdefault("widths", [])
    result.setdefault("longhold", [])
    result.setdefault("schemes", [])
    result.setdefault("repeats", [])
    result.setdefault("threads", [])
    result.setdefault("robust", [])
    result.setdefault("tau", [])
    forcings = [f.strip() for f in args.forcings.split(",") if f.strip()]
    modes = {m.strip() for m in args.mode.split(",") if m.strip()}
    unknown = modes - set(ALL_MODES)
    if unknown:
        raise SystemExit(f"unknown --mode entries {sorted(unknown)}; allowed {sorted(ALL_MODES)}")
    t_start = time.time()

    def on(*names: str) -> bool:
        """True when the mode list selects any of ``names`` (``all`` selects all)."""
        return bool(modes & set(names)) or "all" in modes

    def tiers_to_run():
        if "pilot" in modes:
            return [P.PREREG["target_ratios"][0]]
        return list(P.PREREG["target_ratios"])

    # ---------------------------------------------------------------- Maxwell
    # The Maxwell pairs are cheap (bisection on an analytic EOS) and several arms
    # key off them, so they are always built; they are also the analytic target
    # every ratio-error is measured against.
    need_mw = tiers_to_run()
    for target in need_mw:
        key = f"{target:g}"
        if args.resume and key in result["maxwell_tiers"]:
            continue
        t0 = time.time()
        mw = P.maxwell_at_ratio(target, eos=args.eos)
        mw["wall_s"] = round(time.time() - t0, 1)
        result["maxwell_tiers"][key] = mw
        _save(result_path, result)
        print(
            f"[maxwell] target={target:g} found={mw['found']} T_r={mw.get('T_r')} "
            f"ratio={mw.get('ratio')} rho_l={mw.get('rho_liquid')} rho_v={mw.get('rho_vapour')} "
            f"rho_cross={mw.get('rho_cross')} tol_ok={mw.get('within_target_tol')} ({mw['wall_s']}s)",
            flush=True,
        )

    def do_run(spec: P.RunSpec, mw: dict, store: list) -> dict:
        print(f"[run] {spec.tag} start (steps={spec.steps}, L={spec.L}, device={spec.device})", flush=True)
        rec = P.run_probe(spec, mw)
        rec["code"] = dict(
            driver_sha256=result["driver_sha256"],
            probe_lib_sha256=result["environment"]["probe_lib_sha256"],
            torch=result["environment"]["torch"],
            threads=result["environment"]["threads"],
            machine=result["environment"]["machine"],
        )
        rec["verdict"] = P.classify(rec)
        store.append(rec)
        result["generated_utc"] = datetime.now(timezone.utc).isoformat()
        _save(result_path, result)
        if rec.get("stable"):
            print(
                f"[run] {spec.tag} STABLE steps={rec['steps_run']} ratio_sim={rec['ratio']:.3g} "
                f"(target {rec['ratio_maxwell']:.3g}, err={rec['ratio_error']:.3%}) "
                f"Ma_max={rec['ma_max']:.3e} rho_max={rec['rho_max_field']:.3f} "
                f"mass_rate={rec['mass_rate_per_step']:.2e} cls={rec['verdict']['cls']} "
                f"({rec['wall_s']}s, {rec['ms_per_step']} ms/step)",
                flush=True,
            )
        else:
            print(f"[run] {spec.tag} COLLAPSE at step {rec['collapse_step']}: {rec['error']} ({rec['wall_s']}s)", flush=True)
        return rec

    # ----------------------------------------------------------------- pilot
    if args.mode == "pilot":
        mw = result["maxwell_tiers"][f"{P.PREREG['target_ratios'][0]:g}"]
        rec = do_run(_mk_spec("pilot_ratio10", mw, args, forcing="edm", steps=min(args.steps, 2000), L=64, R=14), mw, result["tiers"])
        print(json.dumps({k: rec.get(k) for k in ("stable", "ratio", "ma_max", "ratio_error")}, indent=1))
        return 0 if rec["stable"] and rec["verdict"]["cls"] in ("PASS", "MARGINAL") else 1

    # --------------------------------------------------------------- schemes
    # Same tiers under the alternative force discretisation (pressure-tensor
    # form: exactly curl-free, momentum-conserving) — a *lever* probe, not a
    # gate retry: the predeclared gates are applied identically.
    if on("schemes", "all"):
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for schem in [x.strip() for x in args.schemes.split(",") if x.strip()]:
                for forcing in forcings:
                    tag = f"tier_ratio{target:g}_{forcing}_scheme-{schem}"
                    if any(r["tag"] == tag for r in result["schemes"]):
                        continue
                    spec = _mk_spec(tag, mw, args, forcing=forcing)
                    spec.scheme = schem
                    do_run(spec, mw, result["schemes"])

    # ----------------------------------------------------------------- tiers
    if on("tiers", "all"):
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for forcing in forcings:
                tag = f"tier_ratio{target:g}_{forcing}_L{args.L}R{args.R:g}"
                if any(r["tag"] == tag for r in result["tiers"]):
                    continue
                do_run(_mk_spec(tag, mw, args, forcing=forcing), mw, result["tiers"])

    # -------------------------------------------------------------- boundary
    if on("boundary", "all"):
        sweep = [float(x) for x in args.tr_sweep.split(",") if x.strip()]
        for tr in sweep:
            mw = P.maxwell_at_tr(args.eos, tr)
            for forcing in forcings:
                tag = f"edge_Tr{tr:.2f}_{forcing}"
                if any(r["tag"] == tag for r in result["boundary"]):
                    continue
                do_run(
                    P.RunSpec(
                        tag=tag,
                        eos=args.eos,
                        target_ratio=None,
                        T_r=tr,
                        L=args.L,
                        R=args.R,
                        forcing=forcing,
                        steps=args.boundary_steps,
                        sample=max(50, args.sample // 2),
                        device=args.device,
                    ),
                    mw,
                    result["boundary"],
                )

    # ---------------------------------------------------------------- widths
    if on("widths", "all"):
        widths = [float(w) for w in args.widths.split(",") if w.strip()]
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for w in widths:
                tag = f"width_ratio{target:g}_w{w:g}_edm"
                if any(r["tag"] == tag for r in result["widths"]):
                    continue
                spec = _mk_spec(tag, mw, args, forcing="edm")
                spec.width = w
                do_run(spec, mw, result["widths"])

    # -------------------------------------------------------------- longhold
    # Long hold arm: separates "not yet relaxed" from "true steady state".  Run at
    # more than one initial interface width, because the width-4 construct sits at
    # the stability edge (see the repeats arm).
    if on("longhold", "all"):
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for w in [float(x) for x in args.longhold_widths.split(",") if x.strip()]:
                tag = f"hold_ratio{target:g}_edm_w{w:g}_{args.longhold_steps}steps"
                if any(r["tag"] == tag for r in result["longhold"]):
                    continue
                spec = _mk_spec(tag, mw, args, forcing="edm", steps=args.longhold_steps)
                spec.sample = max(args.sample, 250)
                spec.width = w
                do_run(spec, mw, result["longhold"])

    # Replication arm: identical configuration, `--repeats` realisations, to
    # measure run-to-run variability at the marginal construction widths.
    if on("repeats", "all"):
        target = P.PREREG["target_ratios"][0]
        mw = result["maxwell_tiers"][f"{target:g}"]
        for w in [float(x) for x in args.repeat_widths.split(",") if x.strip()]:
            for k in range(1, args.repeats + 1):
                tag = f"repeat{k}_ratio{target:g}_edm_w{w:g}"
                if any(r["tag"] == tag for r in result["repeats"]):
                    continue
                spec = _mk_spec(tag, mw, args, forcing="edm")
                spec.width = w
                do_run(spec, mw, result["repeats"])

    # ----------------------------------------------------------- robust arm
    # "Stable for N steps" must not be a property of one trajectory.  A density
    # perturbation far below any physical scale is applied and the survival is
    # re-measured; a usable construct must survive it.
    if on("robust", "all"):
        target = P.PREREG["target_ratios"][0]
        mw = result["maxwell_tiers"][f"{target:g}"]
        for amp in [float(x) for x in args.noise_list.split(",") if x.strip()]:
            for sd in [int(x) for x in args.noise_seeds.split(",") if x.strip()]:
                tag = f"robust_a{amp:g}_seed{sd}_ratio{target:g}_edm_w4"
                if any(r["tag"] == tag for r in result["robust"]):
                    continue
                spec = _mk_spec(tag, mw, args, forcing="edm")
                spec.noise = amp
                spec.seed = sd
                do_run(spec, mw, result["robust"])

    # ---------------------------------------------------------- threads arm
    # Numerical-reproducibility control: the float32 reduction order depends on
    # the thread count, which matters exactly at a marginal construct.  This arm
    # makes that dependence visible instead of leaving an unexplained rerun drift.
    if on("threads", "all"):
        target = P.PREREG["target_ratios"][0]
        mw = result["maxwell_tiers"][f"{target:g}"]
        for nt in [int(x) for x in args.thread_list.split(",") if x.strip()]:
            tag = f"thread_t{nt}_ratio{target:g}_edm_w4"
            if any(r["tag"] == tag for r in result["threads"]):
                continue
            spec = _mk_spec(tag, mw, args, forcing="edm")
            spec.threads = nt
            do_run(spec, mw, result["threads"])

    # ------------------------------------------------------------------ tau
    # Viscosity lever (root-cause discrimination, predeclared in
    # PREREG['tau_arm']).  In a pseudopotential method the spurious current is a
    # balance between the (discretisation-broken) pressure-tensor force and the
    # viscous stress, so the relaxation time tau is a *competing* explanation for
    # the spurious-current gate at a given ratio, independent of the EOS
    # construct.  The arm separates the two failure modes:
    #   * Ma gate tunable by tau but ratio error stuck > 15 %  -> the ratio
    #     maintenance miss is structural (construct/discretisation), and the
    #     spurious gate is a flow-regime statement;
    #   * neither tunable -> the construct dies for a reason tau cannot reach.
    if on("tau", "all"):
        tau_forcings = [f.strip() for f in args.tau_forcings.split(",") if f.strip()]
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for tau in [float(x) for x in args.tau_list.split(",") if x.strip()]:
                for forcing in tau_forcings:
                    tag = f"tau_ratio{target:g}_tau{tau:g}_{forcing}"
                    if any(r["tag"] == tag for r in result["tau"]):
                        continue
                    spec = _mk_spec(tag, mw, args, forcing=forcing)
                    spec.tau = tau
                    do_run(spec, mw, result["tau"])

    # -------------------------------------------------------------- controls
    # Planar-interface control: same EOS point, curvature-free, fully periodic.
    # It measures the coexistence pair the *lattice scheme* selects, which
    # separates discretisation error from droplet geometry / finite domain.
    # Only runs for tier configurations that survived (a control for a
    # collapsed construct answers nothing).
    if on("controls", "all"):
        alive = {(r.get("target_ratio"), r["spec"]["forcing"]) for r in result["tiers"] if r.get("stable")}
        for target in tiers_to_run():
            mw = result["maxwell_tiers"][f"{target:g}"]
            for forcing in forcings:
                if result["tiers"] and (target, forcing) not in alive:
                    continue
                tag = f"planar_ratio{target:g}_{forcing}"
                if any(r["tag"] == tag for r in result["controls"]):
                    continue
                spec = _mk_spec(tag, mw, args, forcing=forcing, geometry="planar", steps=args.control_steps)
                spec.width = args.control_width
                do_run(spec, mw, result["controls"])

    # --------------------------------------------------------------- verdict
    result["verdict"] = _verdict(result)
    result["wall_s_total"] = round(time.time() - t_start, 1)
    _save(result_path, result)
    print("\n=== VERDICT ===")
    print(json.dumps(result["verdict"], indent=1))
    print(f"artefact: {result_path}")
    return 0


def _verdict(result: dict) -> dict:
    tiers = []
    for target in P.PREREG["target_ratios"]:
        rows = [r for r in result["tiers"] if r.get("target_ratio") == target]
        for r in rows:
            tiers.append(
                dict(
                    tag=r["tag"],
                    target_ratio=target,
                    forcing=r["spec"]["forcing"],
                    stable=r["stable"],
                    steps_run=r["steps_run"],
                    collapse_step=r["collapse_step"],
                    ratio_sim=r.get("ratio"),
                    ratio_error=r.get("ratio_error"),
                    ma_max=r.get("ma_max"),
                    u_max=r.get("u_max"),
                    rho_max_field=r.get("rho_max_field"),
                    cls=r["verdict"]["cls"],
                    reason=r["verdict"]["reason"],
                )
            )
    controls = [
        dict(
            tag=r["tag"],
            target_ratio=r.get("target_ratio"),
            stable=r["stable"],
            steps_run=r["steps_run"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            cls=r["verdict"]["cls"],
        )
        for r in result["controls"]
    ]
    widths = [
        dict(
            tag=r["tag"],
            target_ratio=r.get("target_ratio"),
            width=r["spec"]["width"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("widths", [])
    ]
    robust = [
        dict(
            tag=r["tag"],
            noise=r["spec"]["noise"],
            seed=r["spec"]["seed"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("robust", [])
    ]
    threads = [
        dict(
            tag=r["tag"],
            threads=r["spec"]["threads"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            rho_min_field=r.get("rho_min_field"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("threads", [])
    ]
    tau = [
        dict(
            tag=r["tag"],
            target_ratio=r.get("target_ratio"),
            tau=r["spec"]["tau"],
            forcing=r["spec"]["forcing"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("tau", [])
    ]
    repeats = [
        dict(
            tag=r["tag"],
            width=r["spec"]["width"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            rho_min_field=r.get("rho_min_field"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("repeats", [])
    ]
    schemes = [
        dict(
            tag=r["tag"],
            target_ratio=r.get("target_ratio"),
            forcing=r["spec"]["forcing"],
            scheme=r["spec"]["scheme"],
            stable=r["stable"],
            steps_run=r["steps_run"],
            collapse_step=r["collapse_step"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("schemes", [])
    ]
    longhold = [
        dict(
            tag=r["tag"],
            target_ratio=r.get("target_ratio"),
            stable=r["stable"],
            steps_run=r["steps_run"],
            ratio_sim=r.get("ratio"),
            ratio_error=r.get("ratio_error"),
            ma_max=r.get("ma_max"),
            rho_v=r.get("rho_v"),
            rho_v_maxwell=r.get("rho_v_maxwell"),
            cls=r["verdict"]["cls"],
        )
        for r in result.get("longhold", [])
    ]
    n_pass = sum(1 for t in tiers if t["cls"] == "PASS")
    n_fail = sum(1 for t in tiers if t["cls"] == "FAIL")
    return dict(
        tiers=tiers,
        controls=controls,
        widths=widths,
        longhold=longhold,
        schemes=schemes,
        repeats=repeats,
        tau=tau,
        threads=threads,
        robust=robust,
        tiers_pass=n_pass,
        tiers_fail=n_fail,
        highest_passing_ratio=max((t["target_ratio"] for t in tiers if t["cls"] == "PASS"), default=None),
        lowest_failing_ratio=min((t["target_ratio"] for t in tiers if t["cls"] == "FAIL"), default=None),
        note=(
            "PASS/FAIL per predeclared gates in 'prereg'; a collapse is an accepted "
            "honest-FAIL result and is reported with its onset step"
        ),
    )


if __name__ == "__main__":
    sys.exit(main())