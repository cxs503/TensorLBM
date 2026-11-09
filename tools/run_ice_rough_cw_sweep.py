#!/usr/bin/env python3
"""Ice-keel roughness -> ice/water drag closure campaign (mission P103W2B-N02).

Commands
--------
``list``       print the case table (ids, Re_tau, k_s^+, grid, steps)
``worker``     run the cases of this worker (``--worker-index/--workers``)
``aggregate``  collect the case JSONs, fit the three-zone laws, adjudicate
               C1-C4 and write ``artifacts/ice_rough_cw/`` (JSON + Markdown +
               figures)

Every number in the report is produced here from the case files; nothing is
transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.ice_roughness import freedrift, geometry as ice_geom, regime_law as rl, sweep  # noqa: E402

DEFAULT_RUN_DIR = ROOT / "runs" / "ice_rough_cw" / "cases"
DEFAULT_ART_DIR = ROOT / "artifacts" / "ice_rough_cw"


def _cases(args: argparse.Namespace):
    targets = None
    if getattr(args, "targets", None):
        targets = tuple(float(v) for v in args.targets.split(",") if v.strip())
    return sweep.campaign_cases(args.profile, ks_plus_targets=targets)


def cmd_list(args: argparse.Namespace) -> int:
    cases = _cases(args)
    print(f"{'case_id':22s} {'kind':7s} {'Re_tau':>8s} {'ks+~':>7s} {'ks_nodes':>8s} "
          f"{'dy+':>6s} {'steps':>7s} {'cells':>10s} {'GU':>7s}")
    total = 0.0
    for c in cases:
        cells = c.h * c.h * c.nz
        gu = cells * c.steps / 1e9
        total += gu
        print(f"{c.case_id:22s} {c.kind:7s} {c.re_tau:8.1f} {c.ks_plus_target:7.1f} "
              f"{c.ks_nodes:8.1f} {c.re_tau / c.h_eff_estimate:6.1f} {c.steps:7d} {cells:10d} {gu:7.2f}")
    print(f"\n{len(cases)} cases, {total:.1f} G cell-updates total")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    cases = _cases(args)
    mine = [c for i, c in enumerate(cases) if i % args.workers == args.worker_index]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"worker {args.worker_index}/{args.workers}: {len(mine)} cases -> {out}", flush=True)
    failed = []
    for c in mine:
        target = out / f"{c.case_id}.json"
        if target.is_file() and not args.force:
            print(f"skip {c.case_id} (done)", flush=True)
            continue
        print(f"run  {c.case_id} Re_tau={c.re_tau:.1f} steps={c.steps}", flush=True)
        try:
            rec = sweep.run_case(c, out, threads=args.threads, rebuild=args.rebuild)
            print(f"  ok   cf={rec['cf']:.5f} ks+={rec['ks_plus']:.1f} "
                  f"wall={rec['wall_seconds']:.0f}s", flush=True)
        except Exception as exc:  # a diverged case must not kill the sweep
            failed.append({"case_id": c.case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {c.case_id}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    if failed:
        (out / f"failures_worker{args.worker_index}.json").write_text(json.dumps(failed, indent=2))
    return 0 if not failed else 1


def cmd_aggregate(args: argparse.Namespace) -> int:
    summary = sweep.aggregate(args.run_dir)
    art = Path(args.out)
    art.mkdir(parents=True, exist_ok=True)
    (art / "summary.json").write_text(json.dumps(summary, indent=2))
    md = render_markdown(summary)
    (art / "REPORT.md").write_text(md)
    try:
        figure_paths = render_figures(summary, art)
        summary["figures"] = [str(p.relative_to(ROOT)) for p in figure_paths]
        (art / "summary.json").write_text(json.dumps(summary, indent=2))
    except Exception as exc:  # figures are nice to have; the numbers are not
        print(f"figure generation failed: {exc}", file=sys.stderr)
    print(json.dumps({k: summary[k] for k in
                      ("C1_smooth_wall_cf", "C2_fully_rough_vs_nikuradse",
                       "C3_transition_monotone", "all_passed")}, indent=2)[:2000])
    return 0


def render_markdown(s: dict[str, Any]) -> str:
    lines = ["# Ice-keel roughness -> ice/water drag closure (P103W2B-N02)", ""]
    lines.append(f"cases: {s['n_rough']} rough + {s['n_smooth']} smooth; "
                 f"all criteria passed: **{s['all_passed']}**")
    lines += ["", "## Fitted three-zone laws", "",
              "| group | lambda/ks | k0 | k0 band | k_s,eff/k_s | smooth upper | fully rough lower | points | measure |",
              "|---|---|---|---|---|---|---|---|---|"]
    for g in s["groups"]:
        f = g["fit"]
        r = g["regimes"]
        if not math.isfinite(float(g["lambda_over_ks"])):
            lines.append(f"| {g['group']} | n/a | {f['k0']:.3f} | "
                         f"[{f['k0_low']:.2f}, {f['k0_high']:.2f}] | "
                         f"{g['equivalent_sand_grain_ratio']:.2f} | - | - | {f['n_points']} | {f['measure']} |")
            continue
        lines.append(f"| {g['group']} | {g['lambda_over_ks']:g} | {f['k0']:.3f} | "
                     f"[{f['k0_low']:.2f}, {f['k0_high']:.2f}] | "
                     f"{g['equivalent_sand_grain_ratio']:.2f} | "
                     f"{r['ks_plus_smooth_upper']:.1f} | {r['ks_plus_fully_rough_lower']:.1f} | "
                     f"{f['n_points']} | {f['measure']} |")
    c1 = s["C1_smooth_wall_cf"]
    lines += ["", "## C1 smooth-wall Cf", "",
              f"threshold {c1['threshold_rel_error']:.1%}; max |rel err| "
              f"= {c1['max_abs_rel_error']:.2%}; passed **{c1['passed']}**", "",
              "| case | Re_tau | Cf | reference | rel err | dy+ |", "|---|---|---|---|---|---|"]
    for r in c1["rows"]:
        lines.append(f"| {r['case_id']} | {r['re_tau']:.0f} | {r['cf']:.5f} | "
                     f"{r['cf_reference']:.5f} | {r['rel_error']:+.2%} | {r['dy_plus']:.1f} |")
    c2 = s["C2_fully_rough_vs_nikuradse"]
    lines += ["", "## C2 fully rough vs Nikuradse", "",
              f"scope: {c2['scope']}; threshold {c2['threshold_rel_error']:.1%}; max |rel err| "
              f"= {c2['max_abs_rel_error']:.2%}; passed **{c2['passed']}**", "",
              "| case | lambda/ks | ks+ | Cf | Cf Nik | rel err | log window |", "|---|---|---|---|---|---|---|"]
    for r in c2["all_fully_rough"]:
        lines.append(f"| {r['case_id']} | {r['group']} | {r['ks_plus']:.0f} | {r['cf']:.5f} | "
                     f"{r['cf_nikuradse']:.5f} | {r['cf_rel_error_vs_nikuradse']:+.2%} | {r['log_window_valid']} |")
    c3 = s["C3_transition_monotone"]
    lines += ["", f"## C3 monotone transition: passed **{c3['passed']}** "
                  f"({c3['n_points_checked']} points checked)", ""]
    c4 = s["C4_nansen_drift_angle"]
    lines += ["## C4 Nansen drift angle", "",
              f"threshold {c4['threshold_deg']:.0f} deg; passed **{c4['passed']}**"]
    if "backfill" in c4:
        b = c4["backfill"]
        lines += [f"- closure C_w = {b['closure']['C_w']:.5f} -> drift {b['closure']['drift_speed_m_s']:.3f} m/s, "
                  f"angle {b['closure']['drift_angle_deg']:.2f} deg (k_s+ = {b['closure']['ks_plus']:.0f}, "
                  f"{b['closure']['regime']})",
                  f"- constant C_w = {b['constant_C_w']:.4f} -> angle {b['constant']['drift_angle_deg']:.2f} deg",
                  f"- Nansen observation {b['nansen_angle_deg']:.0f} deg (band {b['nansen_band_deg']})",
                  f"- angle error with closure: {c4['angle_error_deg']:.2f} deg"]
    lines += ["", "## Rough cases", "",
              "| case | lambda/ks | ks+ | Re_tau | dy+ | Cf | dU+ | measure | log window |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in s["rough_cases"]:
        lines.append(f"| {r['case_id']} | {r['group']} | {r['ks_plus']:.0f} | {r['re_tau']:.0f} | "
                     f"{r['dy_plus']:.1f} | {r['cf']:.5f} | {r['dU_plus_used']:+.2f} | "
                     f"{r['dU_plus_measure']} | {r['log_window_valid']} |")
    return "\n".join(lines) + "\n"


def render_figures(s: dict[str, Any], art: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths: list[Path] = []
    colors = {2.0: "#c62828", 5.0: "#1565c0", 10.0: "#2e7d32"}

    # --- Fig 1: roughness function ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ks_grid = np.logspace(0, 3.1, 200)
    ax.plot(ks_grid, rl.nikuradse_delta_u_plus(ks_grid), "k--", lw=1.6,
            label="Nikuradse sand grain")
    for g in s["groups"]:
        key = g["group"]
        k0 = g["fit"]["k0"]
        if not math.isfinite(k0):
            continue
        col = "k" if key == "sand" else colors.get(float(key), "gray")
        ax.plot(ks_grid, rl.delta_u_plus(ks_grid, k0), color=col, lw=1.4, ls="-" if key != "sand" else "-.",
                label=f"law {key if key=='sand' else 'lambda/ks='+key} (k0={k0:.2f})")
        rows = [r for r in s["rough_cases"] if r["group"] == key
                and r["log_window_valid"] and r["dU_plus_profile"] == r["dU_plus_profile"]]
        if rows:
            ax.plot([r["ks_plus"] for r in rows], [r["dU_plus_profile"] for r in rows], "o",
                    color=col, ms=5, mfc="white")
        rows_b = [r for r in s["rough_cases"] if r["group"] == key
                  and not r["log_window_valid"] and r["dU_plus_bulk"] == r["dU_plus_bulk"]]
        if rows_b:
            ax.plot([r["ks_plus"] for r in rows_b], [r["dU_plus_bulk"] for r in rows_b], "x",
                    color=col, ms=7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_s^+$")
    ax.set_ylabel(r"$\Delta U^+$")
    ax.set_title("Roughness function: LBM (symbols) vs three-zone law")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    p = art / "fig1_roughness_function.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Fig 2: Cf / Cf_smooth vs ks+ ------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for g in s["groups"]:
        key = g["group"]
        rows = [r for r in s["rough_cases"] if r["group"] == key]
        rows.sort(key=lambda r: r["ks_plus"])
        if not rows:
            continue
        col = "k" if key == "sand" else colors.get(float(key), "gray")
        cf0 = [rl.cf_channel_smooth_reference(r["re_tau"]) for r in rows]
        ax.plot([r["ks_plus"] for r in rows], [r["cf"] / c for r, c in zip(rows, cf0)], "o-",
                color=col, lw=1.2, ms=5, label=f"LBM {key if key=='sand' else 'lambda/ks='+key}")
        if math.isfinite(g["fit"]["k0"]):
            law = rl.ThreeZoneLaw(float(g["lambda_over_ks"]) if key != "sand" else 1.0, g["fit"]["k0"])
            ks = np.array([r["ks_plus"] for r in rows])
            ax.plot(ks, [law.cf_ratio(k, c) for k, c in zip(ks, cf0)], ":", color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k_s^+$"); ax.set_ylabel(r"$C_f / C_{f,\mathrm{smooth}}$")
    ax.set_title("Friction increase (symbols: LBM, dotted: closure)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    p = art / "fig2_cf_ratio.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Fig 3: drift angle sweep ---------------------------------------
    c4 = s.get("C4_nansen_drift_angle", {})
    if c4.get("sweep"):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ua = [r["wind_speed_m_s"] for r in c4["sweep"]]
        ax.plot(ua, [r["closure_angle_deg"] for r in c4["sweep"]], "o-", label="roughness closure")
        if "constant_angle_deg" in c4["sweep"][0]:
            ax.plot(ua, [r["constant_angle_deg"] for r in c4["sweep"]], "s--",
                    label=f"constant C_w = {c4['sweep'][0]['constant_C_w']:.4f}")
        ax.axhline(freedrift.NANSEN_DRIFT_ANGLE_DEG, color="k", ls=":", lw=1.2,
                   label="Nansen observation")
        ax.fill_between(ua, *freedrift.NANSEN_ANGLE_BAND_DEG, color="k", alpha=0.08,
                        label="observed band")
        ax.set_xlabel("wind speed [m/s]"); ax.set_ylabel("drift angle [deg]")
        ax.set_title("Nansen free-drift angle: constant vs roughness C_w")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        p = art / "fig3_nansen_angle.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # --- Fig 4: mean velocity profiles (log-law coordinates) -------------
    prof_dir = Path(s["run_dir"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    samples = [r for r in s["rough_cases"] if r["group"] == "5"]
    for r in sorted(samples, key=lambda r: r["ks_plus"]):
        rec_path = prof_dir / f"{r['case_id']}.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text())
        prof = rec.get("profile") or {}
        yp, up = prof.get("y_plus"), prof.get("u_plus")
        if not yp:
            continue
        ax.plot(yp, up, lw=1.2, label=f"ks+={r['ks_plus']:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$y^+$"); ax.set_ylabel(r"$U^+$")
    ax.set_title(r"Mean profiles, $\lambda/k_s=5$ (LBM)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    p = art / "fig4_profiles.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", default="campaign", choices=sorted(sweep.PROFILES))

    p = sub.add_parser("list"); add_common(p)
    p.add_argument("--targets", default=None,
                   help="comma separated k_s^+ targets (default: the mission sweep)")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("worker"); add_common(p)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--targets", default=None,
                   help="comma separated k_s^+ targets (default: the mission sweep)")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("aggregate")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--out", default=str(DEFAULT_ART_DIR))
    p.set_defaults(func=cmd_aggregate)

    args = ap.parse_args(argv)
    for a in ("worker_index", "workers", "threads"):
        if hasattr(args, a) and a in ("worker_index", "workers") and args.workers < 1:
            ap.error("--workers must be >= 1")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())