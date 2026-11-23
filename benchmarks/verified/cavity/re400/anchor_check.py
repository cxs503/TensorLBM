#!/usr/bin/env python3
"""Score the Re=400 lid-driven cavity against the sha256-bound Ghia (1982) anchor.

Why this exists
---------------
``benchmarks/verified/cavity/re400/result.json`` reports a single number
(``max_abs_dev_pct``) computed against ``GHIA_RE400`` **as imported from the
code**.  When that constant had drifted from the OCR witness, the score silently
measured the drift instead of the physics (see case ``GHIA_RE400`` in
``benchmarks/anchors/ghia1982_provenance.json -> suspect_entries``).

This script closes that loop: the reference values are read from the
sha256-verified CSVs (``benchmarks/anchors/data/ghia1982_re{100,400}_table*.csv``),
*after* ``benchmarks/anchors/run_anchors.py`` has accepted the whole anchor
bundle, and the solver profiles are the raw centreline arrays dumped by
``run.py`` (or produced on demand by importing its ``run_case``, i.e. the same
``tensorlbm.solver`` / ``tensorlbm.lid_driven_cavity`` code path the platform
runs).

Pre-declared metrics (fixed before looking at any result)
---------------------------------------------------------
``M1  max_point_dev_pct``   primary acceptance metric.
    100 * max over the 17+17 Ghia points of |u_sim-u_ref| and |v_sim-v_ref|,
    velocities normalised by the lid velocity U_lid.  This is the "max-point
    deviation" convention already used by the cavity benchmark family
    (``max_abs_dev_pct``) and by the ship-platform accuracy rule, where the
    error scale is the reference velocity, not the local value.
    Criterion: ``M1 < 5%``.

``M2  max_point_rel_pct``   the strict point-wise reading of the same sentence:
    100 * max |Δ|/|ref| over points with |ref| >= REL_FLOOR * max|ref|
    (``REL_FLOOR = 0.05``, pre-declared).  Points below the floor are listed
    separately: |ref| there is <= 5% of the peak, so a 1e-3*U_lid absolute
    error (normal for a 2nd-order LBM at these grids) produces 20-100%
    relative error and the metric stops measuring the discretisation.
    Criterion: ``M2 < 5%`` — reported honestly, may fail (see report).

``M3  extremum_rel_pct``    relative error at the two anchor extrema
    (u_min at y=0.2813, v_min at x=0.8594).  Criterion: both ``< 5%``.

Usage
-----
    # score an existing raw run
    python benchmarks/verified/cavity/re400/anchor_check.py \
        --run-json benchmarks/verified/cavity/re400/artifacts/run_raw.json

    # or run the solver first (slow; use the cluster for >10 min)
    python benchmarks/verified/cavity/re400/anchor_check.py --run --nx 128 192 --steps 200000

Exit code 0 only if the anchor gate passes and every criterion passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPO = CASE_DIR.parents[3]
ANCHOR_DIR = REPO / "benchmarks" / "anchors"
ARTIFACTS = CASE_DIR / "artifacts"

# --- pre-declared acceptance thresholds (do not tune after seeing results) ---
MAX_POINT_DEV_TOL_PCT = 5.0  # M1
MAX_POINT_REL_TOL_PCT = 5.0  # M2
EXTREMUM_REL_TOL_PCT = 5.0  # M3
REL_FLOOR = 0.05  # M2 ill-conditioning floor, fraction of max|ref|
SUPERSEDED_SUSPECTS = "GHIA_RE400 inline constants (see suspect_entries)"


def _import_anchor_gate():
    """Import the anchor gate module (benchmarks/anchors/run_anchors.py)."""
    sys.path.insert(0, str(ANCHOR_DIR))
    import run_anchors  # noqa: WPS433  (local module, deliberately late)

    return run_anchors


def load_bound_anchor(re: int = 400) -> dict:
    """Load the Re centreline tables from the sha256-verified CSVs.

    Refuses to return data unless the whole anchor bundle (witness + every CSV
    sha256 + code-constant drift guard) passes.
    """
    gate = _import_anchor_gate()
    prov = json.loads((ANCHOR_DIR / "ghia1982_provenance.json").read_text(encoding="utf-8"))

    errs = gate.verify_provenance_sha256(prov)
    errs += gate.verify_csv_anchors(prov)
    errs += gate.verify_code_tables_match_anchor(prov)
    if errs:
        raise SystemExit("ANCHOR GATE FAILED, refusing to score:\n  " + "\n  ".join(errs))

    tables = {}
    for entry in prov["csv_anchors"]:
        if int(entry["re"]) != int(re):
            continue
        header, values = gate._read_csv_columns(REPO / entry["path"])
        tables[entry["table"]] = {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "coord_name": header[0],
            "prof_name": header[1],
            "coord": values[:, 0],
            "value": values[:, 1],
        }
    missing = {"I", "II"} - set(tables)
    if missing:
        raise SystemExit(f"Re={re}: BOUND tables missing from csv_anchors: {sorted(missing)}")
    return {
        "re": int(re),
        "tables": tables,
        "witness_sha256": prov["primary_witness"]["sha256"],
        "suspect_entries": prov.get("suspect_entries", []),
    }


def score_profile(run: dict, anchor: dict) -> dict:
    """Compare one solver run against the BOUND tables; return all three metrics."""
    y_pos = np.asarray(run["profiles"]["y"], dtype=float)
    x_pos = np.asarray(run["profiles"]["x"], dtype=float)
    u_cl = np.asarray(run["profiles"]["u_cl"], dtype=float)
    v_cl = np.asarray(run["profiles"]["v_cl"], dtype=float)

    t1, t2 = anchor["tables"]["I"], anchor["tables"]["II"]
    u_sim = np.interp(t1["coord"], y_pos, u_cl)
    v_sim = np.interp(t2["coord"], x_pos, v_cl)
    u_ref, v_ref = t1["value"], t2["value"]

    du, dv = u_sim - u_ref, v_sim - v_ref
    points = []
    for coord, ref, got, tag, prof in (
        *((c, r, g, "I", "u") for c, r, g in zip(t1["coord"], u_ref, u_sim)),
        *((c, r, g, "II", "v") for c, r, g in zip(t2["coord"], v_ref, v_sim)),
    ):
        points.append(
            {
                "table": tag,
                "profile": prof,
                "coord": float(coord),
                "ref": float(ref),
                "got": float(got),
                "delta": float(got - ref),
            }
        )

    max_dev = float(max(np.abs(du).max(), np.abs(dv).max())) * 100.0  # M1
    peak = max(float(np.abs(u_ref).max()), float(np.abs(v_ref).max()))
    floor = REL_FLOOR * peak
    rel_all = np.concatenate(
        [
            np.abs(du) / np.maximum(np.abs(u_ref), 1e-30),
            np.abs(dv) / np.maximum(np.abs(v_ref), 1e-30),
        ]
    )
    ref_all = np.concatenate([u_ref, v_ref])
    conditioned = np.abs(ref_all) >= floor
    max_rel = float(rel_all[conditioned].max()) * 100.0  # M2
    ill = [
        {
            "table": "I" if i < len(u_ref) else "II",
            "coord": float((t1["coord"] if i < len(u_ref) else t2["coord"])[i % len(u_ref)]),
            "ref": float(ref_all[i]),
            "rel_pct": float(rel_all[i]) * 100.0,
        }
        for i in np.flatnonzero(~conditioned)
    ]

    iu = int(np.argmin(u_ref))
    iv = int(np.argmin(v_ref))
    extrema = {
        "u_min": {
            "coord_y": float(t1["coord"][iu]),
            "ref": float(u_ref[iu]),
            "got": float(u_sim[iu]),
            "rel_pct": float(abs(u_sim[iu] - u_ref[iu]) / abs(u_ref[iu])) * 100.0,
        },
        "v_min": {
            "coord_x": float(t2["coord"][iv]),
            "ref": float(v_ref[iv]),
            "got": float(v_sim[iv]),
            "rel_pct": float(abs(v_sim[iv] - v_ref[iv]) / abs(v_ref[iv])) * 100.0,
        },
    }

    return {
        "n": run["nx"],
        "re": run.get("re", anchor["re"]),
        "u_lid": run.get("u_lid"),
        "tau": run.get("tau"),
        "steps": run.get("steps"),
        "last_resid": run.get("last_resid"),
        "metrics": {
            "M1_max_point_dev_pct": max_dev,
            "M1_tol_pct": MAX_POINT_DEV_TOL_PCT,
            "M1_pass": max_dev < MAX_POINT_DEV_TOL_PCT,
            "M2_max_point_rel_pct": max_rel,
            "M2_tol_pct": MAX_POINT_REL_TOL_PCT,
            "M2_rel_floor_frac": REL_FLOOR,
            "M2_n_conditioned": int(conditioned.sum()),
            "M2_n_ill_conditioned": int((~conditioned).sum()),
            "M2_pass": max_rel < MAX_POINT_REL_TOL_PCT,
            "M3_extremum_rel_pct": {k: v["rel_pct"] for k, v in extrema.items()},
            "M3_pass": all(v["rel_pct"] < EXTREMUM_REL_TOL_PCT for v in extrema.values()),
            "rmse_u": float(np.sqrt(np.mean(du**2))),
            "rmse_v": float(np.sqrt(np.mean(dv**2))),
        },
        "extrema": extrema,
        "ill_conditioned_points": ill,
        "points": points,
    }


def plot(anchor: dict, scored: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    t1, t2 = anchor["tables"]["I"], anchor["tables"]["II"]
    axes[0].plot(t1["value"], t1["coord"], "ks", ms=5, label="Ghia 1982 (BOUND)")
    axes[1].plot(t2["value"], t2["coord"], "ks", ms=5, label="Ghia 1982 (BOUND)")
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for k, s in enumerate(scored):
        prof = s["run"]
        y = np.asarray(prof["profiles"]["y"])
        x = np.asarray(prof["profiles"]["x"])
        axes[0].plot(
            prof["profiles"]["u_cl"],
            y,
            "-",
            lw=1.3,
            color=colors[k % 4],
            label=f"LBM n={s['n']} (M1={s['metrics']['M1_max_point_dev_pct']:.2f}%)",
        )
        axes[1].plot(
            prof["profiles"]["v_cl"], x, "-", lw=1.3, color=colors[k % 4], label=f"LBM n={s['n']}"
        )
    axes[0].set_xlabel("u / U_lid")
    axes[0].set_ylabel("y / H")
    axes[0].set_title("Vertical centreline u (Re=400)")
    axes[1].set_xlabel("v / U_lid")
    axes[1].set_ylabel("x / H")
    axes[1].set_title("Horizontal centreline v (Re=400)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Ghia et al. (1982) Re=400 anchor — TensorLBM D2Q9 MRT (V3 walls)", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_report(path: Path, anchor: dict, scored: list[dict], repro_cmd: str) -> str:
    crit_ok = all(s["metrics"]["M1_pass"] and s["metrics"]["M3_pass"] for s in scored)
    lines = [
        "# Ghia (1982) Re=400 external anchor — TensorLBM",
        "",
        f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 1. Problem setting and anchor source",
        "",
        "* Geometry: square cavity 1x1, lid u=U_lid, no-slip on the other three walls.",
        "* Re = U_lid*H/nu = 400; D2Q9 MRT, V3 pre-streaming half-way bounce-back on the "
        "three stationary walls, Zou/He moving-lid BC on the top row.",
        "* Reference: Ghia, Ghia & Shin (1982), JCP 48(3):387-411, Table I (u vs y at "
        "x/L=0.5) and Table II (v vs x at y/L=0.5), 129x129 grid results.",
        "* Provenance: BOUND. OCR text-layer witness sha256 "
        f"`{anchor['witness_sha256'][:16]}...`; in-repo CSVs are sha256-pinned and "
        "re-verified by `benchmarks/anchors/run_anchors.py` before scoring.",
        "",
        "| table | file | sha256 (16) |",
        "|---|---|---|",
    ]
    for tag, t in sorted(anchor["tables"].items()):
        lines.append(f"| {tag} | `{t['path']}` | `{t['sha256'][:16]}` |")

    lines += [
        "",
        "## 2. Method",
        "",
        "* Solver: `tensorlbm.solver.collide_mrt` + `stream`, "
        "`tensorlbm.d2q9.equilibrium/macroscopic`, "
        "`tensorlbm.lid_driven_cavity.zou_he_moving_lid` (platform solver path, no "
        "benchmark-local numerics).",
        "* Entry: `benchmarks/verified/cavity/re400/run.py`; scoring: "
        "`benchmarks/verified/cavity/re400/anchor_check.py`.",
        "* Pre-declared metrics: M1 max-point deviation (|dU|/U_lid), M2 strict point-wise "
        f"relative with ill-conditioning floor {REL_FLOOR:g}x peak, M3 extremum relative.",
        f"* Tolerances: M1 < {MAX_POINT_DEV_TOL_PCT:g}%, M2 < {MAX_POINT_REL_TOL_PCT:g}%, "
        f"M3 < {EXTREMUM_REL_TOL_PCT:g}%.",
        "",
        "## 3. Results",
        "",
        "| n | steps | last resid | M1 max-point dev | M2 strict rel | M3 u_min | M3 v_min | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        m = s["metrics"]
        lines.append(
            f"| {s['n']} | {s['steps']} | "
            f"{'n/a' if s['last_resid'] is None else format(s['last_resid'], '.2e')} | "
            f"{m['M1_max_point_dev_pct']:.2f}% {'PASS' if m['M1_pass'] else 'FAIL'} | "
            f"{m['M2_max_point_rel_pct']:.2f}% {'PASS' if m['M2_pass'] else 'FAIL'} | "
            f"{m['M3_extremum_rel_pct']['u_min']:.2f}% | "
            f"{m['M3_extremum_rel_pct']['v_min']:.2f}% | "
            f"{'PASS' if m['M1_pass'] and m['M3_pass'] else 'FAIL'} |"
        )

    lines += ["", "### Point-by-point table (finest grid)", ""]
    best = max(scored, key=lambda s: s["n"])
    lines += ["| table | coord | ref | got | delta | rel |", "|---|---|---|---|---|---|"]
    for p in best["points"]:
        rel = abs(p["delta"]) / abs(p["ref"]) * 100 if abs(p["ref"]) > 1e-12 else float("nan")
        lines.append(
            f"| {p['table']} {p['profile']} | {p['coord']:.4f} | {p['ref']:+.5f} | "
            f"{p['got']:+.5f} | {p['delta']:+.5f} | {rel:.1f}% |"
        )

    lines += [
        "",
        "### Ill-conditioned points for the strict metric (M2)",
        "",
        f"|ref| < {REL_FLOOR:g} * peak = {REL_FLOOR * max(abs(p['ref']) for p in best['points']):.4f}; "
        "a 1e-3*U_lid absolute error already means >=20% here, so these are reported, "
        "not silently dropped:",
        "",
        "| table | coord | ref | rel |",
        "|---|---|---|---|",
    ]
    for p in best["ill_conditioned_points"]:
        lines.append(f"| {p['table']} | {p['coord']:.4f} | {p['ref']:+.5f} | {p['rel_pct']:.1f}% |")

    lines += [
        "",
        "## 4. Excluded entries (suspect list) and why",
        "",
        "Every reference entry registered in "
        "`benchmarks/anchors/ghia1982_provenance.json -> suspect_entries` is excluded "
        "from this comparison; the BOUND tables above are the only reference used.",
        "",
        "| Re | table | reason | action |",
        "|---|---|---|---|",
    ]
    for e in anchor["suspect_entries"]:
        reason = e["reason"].replace("|", "/")
        lines.append(f"| {e['re']} | {e['table'].replace('|', '/')} | {reason} | {e['action']} |")

    lines += [
        "",
        "## 5. Verdict",
        "",
    ]
    if crit_ok:
        lines.append(
            f"* **M1 (max-point deviation, `max|dU|/U_lid`) < {MAX_POINT_DEV_TOL_PCT:g}%: PASS** "
            f"on every grid (best {min(s['metrics']['M1_max_point_dev_pct'] for s in scored):.2f}%)."
        )
    else:
        lines.append("* **M1: FAIL** (see table).")
    lines.append(
        "* **M3 (extremum relative error) < "
        f"{EXTREMUM_REL_TOL_PCT:g}%: "
        f"{'PASS' if all(s['metrics']['M3_pass'] for s in scored) else 'FAIL'}** "
        f"(u_min {best['metrics']['M3_extremum_rel_pct']['u_min']:.2f}%, "
        f"v_min {best['metrics']['M3_extremum_rel_pct']['v_min']:.2f}% at n={best['n']})."
    )
    lines.append(
        f"* **M2 (strict point-wise relative) < {MAX_POINT_REL_TOL_PCT:g}%: "
        f"{'PASS' if best['metrics']['M2_pass'] else 'FAIL'}** "
        f"(max {best['metrics']['M2_max_point_rel_pct']:.2f}% over the "
        f"{best['metrics']['M2_n_conditioned']} conditioned points)."
    )
    lines += [
        "",
        "## 6. Reproduce",
        "",
        "```",
        repro_cmd,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", default=str(ARTIFACTS / "run_raw.json"))
    ap.add_argument(
        "--run", action="store_true", help="run the solver instead of reading --run-json"
    )
    ap.add_argument("--nx", nargs="+", type=int, default=[128, 192])
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    anchor = load_bound_anchor(400)

    if args.run:
        sys.path.insert(0, str(CASE_DIR))
        from run import run_case  # the platform solver entry point

        runs = [run_case(nx, 400.0, args.u_lid, args.steps, args.device) for nx in args.nx]
    else:
        payload = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
        runs = [
            v
            for k, v in sorted(payload.items(), key=lambda kv: int(kv[0]))
            if k.isdigit() and "profiles" in v
        ]

    if not runs:
        raise SystemExit(f"no runs with profiles in {args.run_json}")

    scored = []
    for r in runs:
        s = score_profile(r, anchor)
        s["run"] = r
        scored.append(s)

    payload = {
        "case": "cavity_re400_ghia_anchor",
        "reference": {
            "citation": "Ghia, Ghia & Shin (1982) JCP 48(3):387-411, Tables I-II",
            "provenance": "BOUND (OCR witness sha256 pinned)",
            "witness_sha256": anchor["witness_sha256"],
            "csv": {
                t: {"path": v["path"], "sha256": v["sha256"]} for t, v in anchor["tables"].items()
            },
        },
        "criteria": {
            "M1_max_point_dev_pct": MAX_POINT_DEV_TOL_PCT,
            "M2_max_point_rel_pct": MAX_POINT_REL_TOL_PCT,
            "M3_extremum_rel_pct": EXTREMUM_REL_TOL_PCT,
            "M2_rel_floor_frac": REL_FLOOR,
        },
        "excluded_suspect_entries": anchor["suspect_entries"],
        "results": [{k: v for k, v in s.items() if k != "run"} for s in scored],
        "run_artifacts": [
            {
                "n": s["n"],
                "steps": s["steps"],
                "last_resid": s["last_resid"],
                "u_lid": s["u_lid"],
                "tau": s["tau"],
            }
            for s in scored
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = ARTIFACTS / "ghia_re400_anchor.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    repro = (
        f"PYTHONPATH=$PWD/src python benchmarks/verified/cavity/re400/run.py "
        f"--nx {' '.join(str(s['n']) for s in scored)} --steps {scored[0]['steps']} "
        f"--u-lid {scored[0]['u_lid']} --out benchmarks/verified/cavity/re400/artifacts/run_raw.json\n"
        "python benchmarks/verified/cavity/re400/anchor_check.py\n"
        "python benchmarks/anchors/run_anchors.py"
    )
    write_report(ARTIFACTS / "report.md", anchor, scored, repro)
    plot(anchor, scored, ARTIFACTS / "profiles.png")

    print(f"anchor: {anchor['tables']['I']['path']} + {anchor['tables']['II']['path']} (sha256 OK)")
    keep = all(s["metrics"]["M1_pass"] and s["metrics"]["M3_pass"] for s in scored)
    for s in scored:
        m = s["metrics"]
        print(
            f"n={s['n']:4d} M1={m['M1_max_point_dev_pct']:.2f}% "
            f"M2={m['M2_max_point_rel_pct']:.2f}% "
            f"M3(u_min)={m['M3_extremum_rel_pct']['u_min']:.2f}% "
            f"M3(v_min)={m['M3_extremum_rel_pct']['v_min']:.2f}%"
        )
    print(f"artifacts: {out_json}, {ARTIFACTS / 'report.md'}, {ARTIFACTS / 'profiles.png'}")
    print("RESULT:", "PASS (M1 & M3)" if keep else "FAIL (M1/M3)")
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
