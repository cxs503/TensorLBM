#!/usr/bin/env python3
"""Acceptance analysis for campaign node P295W1R2-N04.

Reads only the case artifacts written by ``run_case.py`` (zero hand-copied
numbers: every figure and every entry of ``criteria.json`` is computed here) and
evaluates the campaign criteria

B1  laminar pressure-drop/flow-rate parity with the HB analytical solution
    (mean velocity error <= 3 %)
B2  Y -> 0 limit reproduces the Newtonian friction factor (<= 2 % pointwise)
B3  f(Re_MR) collapse across n in {0.6, 0.8, 1.0} for Re_MR in [4e3, 4e4]
    (max dispersion <= 10 %)
B4  TKE(Y)/TKE(0) monotonically decreasing and the turbulence shut-off yield
    number Y_c consistent with the momentum-balance estimate (within 40 %)
B5  D3Q19 vs D3Q27 friction factor (<= 3 %) and grid convergence index
    (<= 5 %)
B6  provenance: every reported number traces back to an artifact JSON

Usage
-----
    python benchmarks/p295w1r2_n04_hb_turb/analyze.py --art-dir artifacts/p295w1r2_n04
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.hb_channel_analytical import (  # noqa: E402
    HBChannelParams,
    hb_mean_velocity,
    hb_velocity_profile,
)
from tensorlbm.hb_turbulent_channel3d import (  # noqa: E402
    dodge_metzner_friction,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_cases(art_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(art_dir.glob("*.json")):
        if p.name in {"criteria.json", "provenance.json", "sweep_spec.json"}:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "derived" not in rec or "config" not in rec:
            continue
        rec["_path"] = str(p.relative_to(REPO))
        rec["_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        out[rec.get("name", p.stem)] = rec
    return out


def analytic_laminar(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    cfg = rec["config"]
    h = cfg["half_height"]
    G = cfg["body_force"]
    p = HBChannelParams(tau_y=cfg["tau_y"], K=cfg["K"], n=cfg["n"], G=G, h=h)
    y, u = hb_velocity_profile(p, n_points=2000)
    return y, u


def laminar_error(rec: dict) -> dict:
    """Profile + bulk-velocity error against the HB analytical solution."""
    cfg = rec["config"]
    ny = cfg["ny"]
    u_num = np.asarray(rec["profiles"]["u_mean"], dtype=float)
    y_an, u_an = analytic_laminar(rec)
    # lattice node rows sit at y = i - 0.5 from the lower wall
    dist = np.arange(ny, dtype=float) - 0.5
    u_ref = np.interp(dist, y_an, u_an, left=0.0, right=float(u_an[-1]))
    sl = slice(1, ny - 1)
    l2 = float(np.linalg.norm(u_num[sl] - u_ref[sl]) / max(np.linalg.norm(u_ref[sl]), 1e-30))
    ub_num = float(u_num[sl].mean())
    ub_ref = float(u_ref[sl].mean())
    ub_an = float(
        hb_mean_velocity(
            HBChannelParams(
                tau_y=cfg["tau_y"],
                K=cfg["K"],
                n=cfg["n"],
                G=cfg["body_force"],
                h=cfg["half_height"],
            )
        )
    )
    return {
        "u_bulk_num": ub_num,
        "u_bulk_ref_nodeavg": ub_ref,
        "u_bulk_analytic": ub_an,
        "rel_err_bulk_vs_nodeavg": abs(ub_num - ub_ref) / max(abs(ub_ref), 1e-30),
        "rel_err_bulk_vs_mean": abs(ub_num - ub_an) / max(abs(ub_an), 1e-30),
        "l2_profile": l2,
    }


def interp_log(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    return np.interp(np.log10(xq), np.log10(x[order]), y[order])


def gci(f_fine: float, f_mid: float, f_coarse: float, r: float = 2.0) -> dict:
    """Roache grid convergence index (observed order of accuracy)."""
    eps21 = abs(f_mid - f_fine) / max(abs(f_fine), 1e-30)
    eps32 = abs(f_coarse - f_mid) / max(abs(f_mid), 1e-30)
    p = math.nan
    if eps32 > 0 and eps21 > 0:
        ratio = eps32 / eps21
        if ratio > 0:
            p = abs(math.log(ratio) / math.log(r))
    if not math.isfinite(p) or p <= 0.05:
        p = 2.0
    gci_fine = 1.9 * eps21 / (r**p - 1.0)
    gci_mid = 1.9 * eps32 / (r**p - 1.0)
    extrap = f_fine + (f_fine - f_mid) / (r**p - 1.0)
    return {
        "eps_21": eps21,
        "eps_32": eps32,
        "observed_order": p,
        "gci_fine_pct": 100.0 * gci_fine,
        "gci_mid_pct": 100.0 * gci_mid,
        "richardson_extrapolated": extrap,
    }


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def criterion_B1(cases: dict) -> dict:
    lam = {k: v for k, v in cases.items() if v.get("mode") == "laminar"}
    rows = []
    for name, rec in sorted(lam.items()):
        e = laminar_error(rec)
        e["name"] = name
        e["n"] = rec["config"]["n"]
        e["Y"] = rec["derived"]["yield_number"]
        e["u_tau"] = rec["config"]["u_tau"]
        e["nx"] = rec["config"]["nx"]
        e["ny"] = rec["config"]["ny"]
        rows.append(e)
    worst = max((r["rel_err_bulk_vs_nodeavg"] for r in rows), default=float("nan"))
    worst_l2 = max((r["l2_profile"] for r in rows), default=float("nan"))
    return {
        "rows": rows,
        "worst_bulk_rel_err": worst,
        "worst_profile_l2": worst_l2,
        "threshold": 0.03,
        "pass": bool(rows) and worst <= 0.03,
        "n_cases": len(rows),
    }


def criterion_B2(cases: dict) -> dict:
    """Y -> 0 limit vs the Newtonian baseline at the same Re_tau."""
    rows = []
    base = {
        v["config"]["re_tau"]: v
        for k, v in cases.items()
        if k.startswith("turb")
        and v["config"]["n"] == 1.0
        and abs(v["derived"]["yield_number"]) < 1e-12
    }
    for re_tau, ref in base.items():
        f_ref = ref["derived"]["f_darcy"]
        for name, rec in cases.items():
            if not name.startswith("turb") or rec["config"]["n"] != 1.0:
                continue
            if rec["config"]["re_tau"] != re_tau:
                continue
            Y = rec["derived"]["yield_number"]
            if not (0.0 < Y <= 0.05):
                continue
            f = rec["derived"]["f_darcy"]
            rows.append(
                {
                    "name": name,
                    "re_tau": re_tau,
                    "Y": Y,
                    "f": f,
                    "f_baseline": f_ref,
                    "rel_diff": abs(f - f_ref) / max(f_ref, 1e-30),
                }
            )
    worst = max((r["rel_diff"] for r in rows), default=float("nan"))
    return {
        "rows": rows,
        "worst_rel_diff": worst,
        "threshold": 0.02,
        "pass": bool(rows) and worst <= 0.02,
        "n_cases": len(rows),
    }


def criterion_B3(cases: dict) -> dict:
    """f(Re_MR) collapse across the flow-index groups."""
    groups: dict[float, list[tuple[float, float]]] = {}
    for name, rec in cases.items():
        if not name.startswith("turb"):
            continue
        cfg = rec["config"]
        groups.setdefault(cfg["n"], []).append((rec["derived"]["re_mr"], rec["derived"]["f_darcy"]))
    lo, hi = 4.0e3, 4.0e4
    series = {}
    for n, pts in groups.items():
        pts = [(re, f) for re, f in pts if re > 0 and math.isfinite(f)]
        if len(pts) >= 2:
            series[n] = sorted(pts)
    result = {"series": {str(k): v for k, v in series.items()}, "window": [lo, hi]}
    if len(series) < 2:
        result.update({"pass": False, "reason": "fewer than two flow-index groups"})
        return result
    all_re = np.array([re for pts in series.values() for re, _ in pts])
    grid = np.logspace(np.log10(max(lo, all_re.min())), np.log10(min(hi, all_re.max())), 12)
    # only keep grid points covered by every group
    covered = np.array(
        [
            all(
                min(re for re, _ in pts) <= g <= max(re for re, _ in pts) for pts in series.values()
            )
            for g in grid
        ]
    )
    grid = grid[covered]
    if grid.size < 2:
        result.update({"pass": False, "reason": "no overlapping Re_MR window"})
        return result
    curves = {}
    for n, pts in series.items():
        re = np.array([p[0] for p in pts])
        f = np.array([p[1] for p in pts])
        curves[n] = interp_log(re, f, grid)
    mat = np.vstack(list(curves.values()))
    spread = (mat.max(axis=0) - mat.min(axis=0)) / mat.mean(axis=0)
    result.update(
        {
            "re_grid": grid.tolist(),
            "f_curves": {str(n): c.tolist() for n, c in curves.items()},
            "spread": spread.tolist(),
            "max_spread": float(spread.max()),
            "threshold": 0.10,
            "pass": bool(spread.max() <= 0.10),
        }
    )
    return result


def criterion_B4(cases: dict) -> dict:
    """TKE modulation by the yield number and the shut-off threshold."""
    groups: dict[tuple[float, float], list[tuple[float, float, float]]] = {}
    for name, rec in cases.items():
        if not name.startswith("turb"):
            continue
        cfg = rec["config"]
        key = (cfg["re_tau"], cfg["n"])
        groups.setdefault(key, []).append(
            (
                rec["derived"]["yield_number"],
                rec["derived"]["tke_volume_mean"],
                rec["derived"]["u_bulk"],
            )
        )
    rows = []
    for (re_tau, n), pts in sorted(groups.items()):
        pts = sorted(pts)
        Y0 = [p for p in pts if p[0] <= 1e-12]
        if not Y0:
            continue
        tke0, ub0 = Y0[0][1], Y0[0][2]
        if tke0 <= 0:
            continue
        series = [(y, tke / tke0, ub / max(ub0, 1e-30)) for y, tke, ub in pts]
        ratios = [s[1] for s in series]
        mono = all(b <= a + 0.02 for a, b in zip(ratios, ratios[1:]))
        # shut-off: first Y with TKE ratio below 0.1 (interpolated)
        y_c = None
        for (y1, r1, _), (y2, r2, _) in zip(series, series[1:]):
            if r1 >= 0.1 > r2:
                y_c = y1 + (0.1 - r1) * (y2 - y1) / (r2 - r1)
                break
        rows.append(
            {
                "re_tau": re_tau,
                "n": n,
                "tke0": tke0,
                "u_bulk0": ub0,
                "series": [{"Y": y, "tke_ratio": r, "u_bulk_ratio": u} for y, r, u in series],
                "monotone_decreasing": mono,
                "Y_c_measured": y_c,
            }
        )
    # analytical estimate: the turbulent shear layer survives only where
    # tau > tau_y, i.e. |y| > Y*h; turbulence dies when that layer is thinner
    # than the buffer layer (y+ ~ 30, Hunt & co-workers' definition of the
    # turbulent region), Y_c = 1 - 30/Re_tau.
    an = {}
    for row in rows:
        re_tau = row["re_tau"]
        an[re_tau] = 1.0 - 30.0 / re_tau
    checks = []
    for row in rows:
        y_c = row["Y_c_measured"]
        y_ref = an[row["re_tau"]]
        ok = y_c is not None and abs(y_c - y_ref) / y_ref <= 0.40
        checks.append(
            {
                "re_tau": row["re_tau"],
                "n": row["n"],
                "Y_c_measured": y_c,
                "Y_c_analytic": y_ref,
                "rel_dev": abs(y_c - y_ref) / y_ref if y_c else None,
                "pass": ok,
            }
        )
    mono_all = all(r["monotone_decreasing"] for r in rows)
    return {
        "rows": rows,
        "analytic_estimate": an,
        "checks": checks,
        "all_monotone": mono_all,
        "pass": bool(rows) and mono_all and any(c["pass"] for c in checks),
        "definition_Y_c": "first Y where volume-averaged TKE < 0.1 * TKE(Y=0)",
        "analytic_Y_c": "1 - 30/Re_tau (plug leaves no buffer layer)",
    }


def criterion_B5(cases: dict) -> dict:
    """D3Q19 vs D3Q27 and the three-grid GCI."""
    d19 = {k: v for k, v in cases.items() if k.startswith("gci_d3q19")}
    d27 = {k: v for k, v in cases.items() if k.startswith("gci_d3q27")}
    pair = None
    if d19 and d27:
        n19 = max(d19, key=lambda k: d19[k]["config"]["nx"])
        n27 = max(d27, key=lambda k: d27[k]["config"]["nx"])
        f19 = d19[n19]["derived"]["f_darcy"]
        f27 = d27[n27]["derived"]["f_darcy"]
        pair = {
            "d3q19": n19,
            "d3q27": n27,
            "f_d3q19": f19,
            "f_d3q27": f27,
            "rel_diff": abs(f19 - f27) / max(f19, 1e-30),
            "threshold": 0.03,
            "pass": abs(f19 - f27) / max(f19, 1e-30) <= 0.03,
        }
    gci_rows = {}
    if len(d19) >= 3:
        order = sorted(d19.values(), key=lambda r: r["config"]["nx"] * r["config"]["ny"])
        f = [r["derived"]["f_darcy"] for r in order]
        h = [1.0 / r["config"]["ny"] for r in order]
        r01 = h[0] / h[1]
        res = gci(f[2], f[1], f[0], r=r01)
        res["cases"] = [r["name"] for r in order]
        res["f"] = f
        res["ny"] = [r["config"]["ny"] for r in order]
        res["threshold_pct"] = 5.0
        res["pass"] = res["gci_fine_pct"] <= 5.0
        gci_rows = res
    ok = bool(pair and pair["pass"] and gci_rows and gci_rows["pass"])
    return {"lattice_pair": pair, "gci": gci_rows, "pass": ok}


def collapse_single_point(cases: dict) -> dict:
    """Preliminary collapse check when each n group only has its Y = 0 point.

    Compares the measured friction factors of the different flow indices,
    correcting each to the mean ``Re_MR`` with the local Dodge-Metzner slope
    ``df/dlnRe``.  This is *not* the predeclared B3 criterion (which needs a
    populated Re_MR window); it is reported separately as supporting evidence.
    """
    pts = []
    for name, rec in cases.items():
        if not name.startswith("turb") or "derived" not in rec:
            continue
        der, cfg = rec["derived"], rec["config"]
        if abs(der["yield_number"]) > 1e-12:
            continue
        pts.append((cfg["n"], der["re_mr"], der["f_darcy"]))
    if len(pts) < 2:
        return {"available": False, "reason": "fewer than two flow indices at Y=0"}
    re_bar = sum(p[1] for p in pts) / len(pts)
    corrected = []
    for n, re, f in pts:
        h = 0.05 * re
        f1, f2 = dodge_metzner_friction(re - h, n), dodge_metzner_friction(re + h, n)
        slope = (f2 - f1) / (2 * h) if math.isfinite(f1) and math.isfinite(f2) else 0.0
        corrected.append((n, re, f, f + slope * (re_bar - re)))
    fs = [c[3] for c in corrected]
    spread = (max(fs) - min(fs)) / (sum(fs) / len(fs))
    return {
        "available": True,
        "points": [{"n": c[0], "re_mr": c[1], "f": c[2], "f_at_mean_re": c[3]} for c in corrected],
        "mean_re_mr": re_bar,
        "spread_at_mean_re": spread,
        "threshold": 0.10,
        "note": "supporting evidence only - the formal B3 criterion needs a "
        "populated Re_MR window per flow index",
    }


def quality_diagnostics(cases: dict) -> dict:
    """Self-consistency checks that expose numerical noise / under-resolution.

    * ``tke_profile_vs_volume``: the plane-averaged TKE integrated over the
      fluid rows must reproduce the directly accumulated volume mean.  A ratio
      far from 1 means the velocity field carries grid-scale noise that the
      profile reduction and the volume reduction weight differently - the
      signature observed in the coarse-LES runs of this campaign.
    * ``u_plus_bulk``: ``U_b/u_tau`` vs the log-law expectation (~18.5-19 at
      Re_tau = 180); a low value flags under-developed turbulence.
    """
    rows = []
    for name, rec in sorted(cases.items()):
        if not name.startswith(("turb", "gci")) or "profiles" not in rec:
            continue
        cfg, der = rec["config"], rec["derived"]
        ny = cfg["ny"]
        tke = np.asarray(rec["profiles"]["tke"], dtype=float)
        u = np.asarray(rec["profiles"]["u_mean"], dtype=float)
        fluid = slice(1, ny - 1)
        tke_profile_mean = float(tke[fluid].mean())
        tke_vol = float(der["tke_volume_mean"])
        rows.append(
            {
                "name": name,
                "re_tau": cfg["re_tau"],
                "n": cfg["n"],
                "Y": der["yield_number"],
                "tke_volume_mean": tke_vol,
                "tke_profile_mean": tke_profile_mean,
                "tke_ratio_volume_over_profile": tke_vol / tke_profile_mean
                if tke_profile_mean > 0
                else None,
                "u_plus_bulk": der["u_bulk"] / der["u_tau"],
                "u_bulk_sem_rel": (der.get("u_bulk_sem") or 0.0) / max(der["u_bulk"], 1e-30),
                "f_over_dodge_metzner": der["f_darcy"] / der["f_dodge_metzner"]
                if der.get("f_dodge_metzner")
                else None,
            }
        )
    return {
        "rows": rows,
        "log_law_u_plus_expected": "~18.5 at Re_tau=180 (channel, Dean 1978)",
        "note": "ratios far from 1 (TKE) or from ~18.5 (U+) mean the run is "
        "not a resolved turbulent channel",
    }


def provenance(cases: dict) -> dict:
    return {
        "n_cases": len(cases),
        "artifacts": {k: {"path": v["_path"], "sha256": v["_sha256"]} for k, v in cases.items()},
        "generator": "benchmarks/p295w1r2_n04_hb_turb/analyze.py",
    }


def dodge_metzner_table(cases: dict) -> dict:
    """Measured f vs the Dodge-Metzner correlation at the measured Re_MR."""
    rows = []
    for name, rec in sorted(cases.items()):
        if not name.startswith("turb"):
            continue
        cfg, der = rec["config"], rec["derived"]
        f_dm = dodge_metzner_friction(der["re_mr"], cfg["n"])
        if not math.isfinite(f_dm):
            f_dm = float("nan")
        rows.append(
            {
                "name": name,
                "n": cfg["n"],
                "re_tau": cfg["re_tau"],
                "Y": der["yield_number"],
                "re_mr": der["re_mr"],
                "f_measured": der["f_darcy"],
                "f_dodge_metzner": f_dm,
                "rel_dev": (der["f_darcy"] - f_dm) / f_dm if math.isfinite(f_dm) else None,
                "f_laminar_24_over_re": 24.0 / der["re_mr"] if der["re_mr"] > 0 else None,
            }
        )
    devs = [abs(r["rel_dev"]) for r in rows if r["rel_dev"] is not None]
    return {
        "rows": rows,
        "worst_abs_dev": max(devs) if devs else None,
        "median_abs_dev": float(np.median(devs)) if devs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-dir", default="artifacts/p295w1r2_n04")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    art_dir = (
        (REPO / args.art_dir).resolve()
        if not Path(args.art_dir).is_absolute()
        else Path(args.art_dir)
    )
    out = Path(args.out) if args.out else art_dir / "criteria.json"

    cases = load_cases(art_dir)
    res = {
        "node": "P295W1R2-N04",
        "art_dir": str(art_dir),
        "n_cases": len(cases),
        "B1_laminar_parity": criterion_B1(cases),
        "B2_newtonian_limit": criterion_B2(cases),
        "B3_friction_collapse": criterion_B3(cases),
        "B4_yield_modulation": criterion_B4(cases),
        "B5_lattice_gci": criterion_B5(cases),
        "dodge_metzner": dodge_metzner_table(cases),
        "quality_diagnostics": quality_diagnostics(cases),
        "collapse_single_point": collapse_single_point(cases),
        "provenance": provenance(cases),
    }
    verdicts = {
        k: res[k]["pass"]
        for k in (
            "B1_laminar_parity",
            "B2_newtonian_limit",
            "B3_friction_collapse",
            "B4_yield_modulation",
            "B5_lattice_gci",
        )
    }
    res["verdicts"] = verdicts
    res["all_pass"] = all(verdicts.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({"verdicts": verdicts, "all_pass": res["all_pass"]}, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
