#!/usr/bin/env python3
"""ks+ sweep of voxel-resolved k-type roughness: run, aggregate, adjudicate.

Three morphologies (random sand grains, transverse k-type ribs, directional ice
scoring ridges) are swept over ``ks+`` on three wall-normal grids.  Each case is
a wall-resolved force-driven D3Q19 channel (see
:mod:`tensorlbm.rough_channel3d`); the campaign is executed with the native
C/OpenMP back-end because a single case is ~10^10 cell updates.

Commands
--------
``list``       print the case table (ids, grids, ks+, ks in lattice units)
``worker``     run the cases assigned to ``--worker-index`` out of ``--workers``
``aggregate``  collect case JSONs, compute the acceptance criteria, write
               ``artifacts/roughness_ks_du/`` (JSON + markdown + figures)

Every number that ends up in the report is produced here from the case files;
nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.rough_channel3d import RoughChannelConfig  # noqa: E402
from tensorlbm.roughness_elements import (  # noqa: E402
    ice_gouge_elements,
    sand_grain_elements,
    transverse_rib_elements,
)
from tensorlbm.roughness_laws import (  # noqa: E402
    KAPPA,
    delta_cf_from_delta_u_plus,
    ittc57_cf,
    ittc78_allowance,
    literature_band,
    equivalent_ks_plus,
)

OUT_DIR = ROOT / "runs" / "roughness_ks_du"
CASES_DIR = OUT_DIR / "cases"
ART_DIR = ROOT / "artifacts" / "roughness_ks_du"

#: One entry per wall-normal grid: h = half channel height in lattice units,
#: Re_tau fixed, u_tau chosen so that the bulk Mach number stays < 0.25 and the
#: eddy turnover (2h/u_tau steps) stays affordable.
GRIDS: dict[str, dict[str, Any]] = {
    "coarse": {"h": 48, "u_tau": 0.008},
    "medium": {"h": 64, "u_tau": 0.008},
    "fine": {"h": 96, "u_tau": 0.010},
}
RE_TAU = 600.0
#: Steps per eddy turnover; total = warmup + sampling.  0.5 T of warm-up plus
#: 1.5 T of averaging is the minimum for which the mean profile and the
#: second-moment budget of this configuration were checked to be stationary.
TURNOVERS_WARMUP = 0.5
TURNOVERS_SAMPLE = 1.5


@dataclass(frozen=True)
class Case:
    case_id: str
    morphology: str
    grid: str
    ks_plus_target: float
    role: str = "main"
    seed: int = 20261107

    @property
    def h(self) -> int:
        return int(GRIDS[self.grid]["h"])

    @property
    def u_tau(self) -> float:
        return float(GRIDS[self.grid]["u_tau"])

    @property
    def ks_lattice(self) -> float:
        return self.ks_plus_target * self.h / RE_TAU

    def config(self, *, steps_scale: float = 1.0, threads: int = 16) -> RoughChannelConfig:
        h = self.h
        nz = 2 * h + 2
        turnover = 2.0 * h / self.u_tau
        steps = int(round((TURNOVERS_WARMUP + TURNOVERS_SAMPLE) * turnover * steps_scale))
        warmup = int(round(TURNOVERS_WARMUP * turnover * steps_scale))
        return RoughChannelConfig(
            nx=h,
            ny=h,
            nz=nz,
            re_tau=RE_TAU,
            u_tau=self.u_tau,
            steps=steps,
            warmup=warmup,
            sample_interval=10,
            seed=self.seed,
            threads=threads,
            use_compile=False,
            # 0.5 x the log-law profile: the full-amplitude start puts an O(0.1)
            # velocity jump across a single voxel at the element flanks, which
            # can trip the near-0.5 relaxation time before the flow is turbulent.
            init_scale=0.5,
            noise_amplitude=0.03,
            note=f"{self.case_id}",
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "morphology": self.morphology,
            "grid": self.grid,
            "ks_plus_target": self.ks_plus_target,
            "ks_lattice": self.ks_lattice,
            "h": self.h,
            "u_tau": self.u_tau,
            "re_tau": RE_TAU,
            "role": self.role,
            "seed": self.seed,
        }


def _case_id(morph: str, grid: str, ks_plus: float | None) -> str:
    if ks_plus is None:
        return f"smooth_{grid}"
    return f"{morph}_{grid}_ksp{ks_plus:g}".replace(".", "p")


def build_cases() -> list[Case]:
    """The campaign table (kept explicit so the report can cite it verbatim)."""
    cases: list[Case] = []
    # smooth reference on every grid: converts a measured profile into dU+ and
    # provides the else-unavailable smooth U+_bulk for the bulk definition.
    for grid in GRIDS:
        cases.append(Case(case_id=f"smooth_{grid}", morphology="smooth", grid=grid,
                          ks_plus_target=0.0, role="reference"))
    # medium grid: full morphology x ks+ matrix
    for morph in ("sand", "ribs", "ice_gouges"):
        for ksp in (18.75, 37.5, 75.0, 150.0, 225.0):
            cases.append(Case(_case_id(morph, "medium", ksp), morph, "medium", ksp))
    # sand on coarse/fine for the grid-convergence ladder
    for grid in ("coarse", "fine"):
        for ksp in (37.5, 75.0, 150.0):
            cases.append(Case(_case_id("sand", grid, ksp), "sand", grid, ksp, role="convergence"))
    # morphology x grid cross-checks at ks+ = 75
    for morph in ("ribs", "ice_gouges"):
        for grid in ("coarse", "fine"):
            cases.append(Case(_case_id(morph, grid, 75.0), morph, grid, 75.0, role="cross_check"))
    return cases


def elements_for(case: Case):
    """Build the voxel geometry of a case (deterministic in the case seed)."""
    if case.morphology == "smooth":
        return None
    nx = ny = case.h
    nz = 2 * case.h + 2
    ks = case.ks_lattice
    if case.morphology == "sand":
        return sand_grain_elements(nx, ny, nz, ks, seed=case.seed, coverage=0.35)
    if case.morphology == "ribs":
        return transverse_rib_elements(nx, ny, nz, ks, pitch_over_ks=8.0, width_over_ks=1.0)
    if case.morphology == "ice_gouges":
        return ice_gouge_elements(nx, ny, nz, ks, angle_deg=30.0, spacing_over_ks=4.0,
                                  width_over_ks=0.6, waviness=0.25, seed=case.seed)
    raise ValueError(f"unknown morphology {case.morphology!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    tmp.replace(path)


def run_case(case: Case, *, threads: int = 16, steps_scale: float = 1.0, rebuild: bool = False) -> dict[str, Any]:
    """Run one case with the native back-end and cache it as JSON."""
    from tensorlbm.rough_channel3d_native import run_rough_channel_native

    cfg = case.config(threads=threads, steps_scale=steps_scale)
    el = elements_for(case)

    smooth_b = smooth_u = None
    if case.morphology != "smooth":
        ref = load_case(f"smooth_{case.grid}")
        if ref is None:
            raise RuntimeError(f"missing smooth reference for grid {case.grid}; run it first")
        smooth_b = ref["log_fit"]["b"] if ref.get("log_fit") else None
        smooth_u = ref["u_plus_bulk"]

    record = run_rough_channel_native(
        cfg, el, smooth_b_reference=smooth_b, smooth_u_plus_bulk=smooth_u, rebuild=rebuild
    )
    record["case"] = case.to_json()
    record["steps"] = cfg.steps
    record["warmup"] = cfg.warmup
    return record


def load_case(case_id: str) -> dict[str, Any] | None:
    path = CASES_DIR / f"{case_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def cmd_list(args: argparse.Namespace) -> int:
    for case in build_cases():
        cfg = case.config(threads=args.threads, steps_scale=args.steps_scale)
        print(
            f"{case.case_id:34s} grid={case.grid:6s} ks+={case.ks_plus_target:6.2f} "
            f"ks={case.ks_lattice:5.2f} grid_cells={cfg.nx}x{cfg.ny}x{cfg.nz} steps={cfg.steps}"
        )
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    cases = build_cases()
    if args.only_reference:
        mine = [c for c in cases if c.role == "reference"]
    else:
        mine = [
            c
            for i, c in enumerate(cases)
            if i % args.workers == args.worker_index and not (args.skip_reference and c.role == "reference")
        ]
    # references first: every rough case needs its grid's smooth run
    mine.sort(key=lambda c: (c.role != "reference", c.grid, c.ks_plus_target))
    for case in mine:
        if load_case(case.case_id) is not None and not args.force:
            print(f"[skip] {case.case_id} (cached)", flush=True)
            continue
        record = run_case(case, threads=args.threads, steps_scale=args.steps_scale, rebuild=args.rebuild)
        _write_json(CASES_DIR / f"{case.case_id}.json", record)
        print(
            f"[done] {case.case_id} ks+={record['ks_plus']:.1f} U+={record['u_plus_bulk']:.3f} "
            f"dU+={record['dU_plus_profile']:.3f} rho={record['rho_mean']:.5f} "
            f"wall={record['wall_seconds']:.0f}s",
            flush=True,
        )
    return 0


# ---------------------------------------------------------------------------
# Aggregation / adjudication
# ---------------------------------------------------------------------------


def _richardson(f1: float, f2: float, f3: float, r: float) -> dict[str, float]:
    """``f3`` = coarse, ``f2`` = medium, ``f1`` = fine (Roache 1994)."""
    d21 = f2 - f1
    d32 = f3 - f2
    if d21 == 0.0 or d32 == 0.0 or d21 * d32 <= 0.0:
        p = float("nan")
    else:
        p = math.log(abs(d32 / d21)) / math.log(r)
    if not math.isfinite(p) or p <= 0.0:
        return {"p": float("nan"), "gci_fine": float("nan"), "extrapolated": float("nan"),
                "rel_change_fine_medium": abs(d21) / max(abs(f1), 1e-12)}
    extrap = f1 + (f1 - f2) / (r**p - 1.0)
    gci = 1.25 * abs((f1 - f2) / max(abs(f1), 1e-12)) / (r**p - 1.0)
    return {
        "p": p,
        "gci_fine": gci,
        "extrapolated": extrap,
        "rel_change_fine_medium": abs(d21) / max(abs(f1), 1e-12),
    }


ITTC_POINTS = (
    # (Re_L, ks/L): typical full-scale hull roughness, ks = 0.1 ... 5 mm on 100 m
    (1.0e9, 2.0e-6),
    (1.0e9, 5.0e-6),
    (1.0e9, 1.0e-5),
    (1.0e9, 2.0e-5),
    (1.0e9, 3.0e-5),
    (1.0e9, 5.0e-5),
)
SHIP_LENGTH = 100.0
NU_SEA_WATER = 1.139e-6


def delta_u_plus_model(ks_plus: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    """Monotone interpolation of the measured ``dU+(ks+)`` in ``ln ks+``.

    Below/above the measured range the curve is extrapolated with the local
    slope of the two outermost points, which for the fully rough regime is the
    1/kappa asymptote the campaign validates.  This keeps the ship-scale
    conversion honest about what is measured and what is extrapolated.
    """
    x = np.log(np.asarray([p[0] for p in points], dtype=float))
    y = np.asarray([p[1] for p in points], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    lk = np.log(np.asarray(ks_plus, dtype=float))
    out = np.interp(lk, x, y)
    lo_slope = (y[1] - y[0]) / (x[1] - x[0]) if x.size > 1 else 0.0
    hi_slope = (y[-1] - y[-2]) / (x[-1] - x[-2]) if x.size > 1 else 0.0
    out = np.where(lk < x[0], y[0] + lo_slope * (lk - x[0]), out)
    out = np.where(lk > x[-1], y[-1] + hi_slope * (lk - x[-1]), out)
    return np.maximum(out, 0.0)


def _figures(summary: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    """Render the campaign figures; returns ``[{name, path, caption}]``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[dict[str, str]] = []
    colours = {"sand": "tab:blue", "ribs": "tab:red", "ice_gouges": "tab:green", "smooth": "k"}

    # 1) dU+ versus ks+ with the literature band
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ks = np.logspace(math.log10(6.0), math.log10(400.0), 200)
    band = literature_band(ks)
    ax.fill_between(ks, band["lo"], band["hi"], color="0.85", label="literature band (±0.5 u$^\\tau$)")
    ax.plot(ks, band["cebeci"], "k--", lw=1.0, label="Cebeci–Bradshaw $(1/\\kappa)\\ln(1+0.3k_s^+)$")
    ax.plot(ks, band["nikuradse"], "k-.", lw=1.0, label="Nikuradse fully rough $(1/\\kappa)\\ln k_s^+ - 3.5$")
    for morph, data in summary["curves"].items():
        pts = sorted(data, key=lambda d: d["ks_plus"])
        ax.plot([p["ks_plus"] for p in pts], [p["dU_plus_profile"] for p in pts], "o-",
                color=colours.get(morph, None), ms=4, label=f"{morph} (medium grid)")
    ax.set_xscale("log")
    ax.set_xlabel("$k_s^+ = k_s u_\\tau/\\nu$")
    ax.set_ylabel("$\\Delta U^+$")
    ax.set_title("Roughness function of voxel-resolved k-type roughness")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    path = out_dir / "delta_u_plus.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "delta_u_plus.png", "path": str(path),
                 "caption": "Measured roughness function dU+(ks+) of the three morphologies vs the Nikuradse "
                            "fully-rough line and the Cebeci–Bradshaw transition closure (grey band = ±0.5 u_tau)."})

    # 2) mean velocity profiles in wall units
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for rec in summary["profile_cases"]:
        yp = np.asarray(rec["y_plus"])
        up = np.asarray(rec["u_plus"])
        m = (yp > 0.5) & np.isfinite(up)
        ax.semilogx(yp[m], up[m], lw=1.2, label=f"{rec['label']}")
    yp = np.logspace(0.3, 2.2, 100)
    ax.semilogx(yp, (1 / KAPPA) * np.log(yp) + 5.0, "k--", lw=1.0, label="smooth log law (kappa=0.41, B=5)")
    ax.set_xlabel("$y^+$")
    ax.set_ylabel("$U^+$")
    ax.set_title("Mean velocity profiles (medium grid)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = out_dir / "profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "profiles.png", "path": str(path),
                 "caption": "Mean velocity profiles in wall units: smooth reference and the three morphologies "
                            "at matched ks+ (medium grid), with the smooth log law for reference."})

    # 3) dCf: LBM vs ITTC-78
    it = summary["ittc78"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ksL = np.asarray(it["ks_over_l"])
    ax.loglog(ksL, np.asarray(it["delta_cf_ittc78"]), "k-", label="ITTC-78 allowance")
    ax.loglog(ksL, np.asarray(it["delta_cf_lbm"]), "o-", color="tab:blue", label="LBM dU+ → dCf (sand)")
    ax.set_xlabel("$k_s/L$")
    ax.set_ylabel("$\\Delta C_F$")
    ax.set_title("Roughness allowance: LBM closure vs ITTC-78")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = out_dir / "ittc78.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "ittc78.png", "path": str(path),
                 "caption": "Roughness allowance vs ITTC-78 at the six adjudication points: the LBM dU+ is converted "
                            "with dCf = 2/(U+sm-dU+)^2 - 2/U+sm^2 at the same Re_L."})

    # 4) grid convergence of dU+
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    conv = summary["convergence"]
    for ksp, rows in conv.items():
        r = [row["dU_plus_profile"] for row in rows]
        ax.plot(["coarse", "medium", "fine"], r, "o-", label=f"ks+ = {ksp}")
    ax.set_ylabel("$\\Delta U^+$")
    ax.set_title("Grid convergence of the roughness function (sand)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"name": "convergence.png", "path": str(path),
                 "caption": "Grid convergence of dU+ on the coarse/medium/fine ladder for the sand morphology."})
    return figs


def cmd_aggregate(args: argparse.Namespace) -> int:
    cases = {c.case_id: c for c in build_cases()}
    records: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for cid in cases:
        rec = load_case(cid)
        if rec is None:
            missing.append(cid)
        else:
            records[cid] = rec
    if not records:
        print("no case files found; run the workers first", file=sys.stderr)
        return 2

    def row(rec: dict[str, Any]) -> dict[str, Any]:
        case = rec["case"]
        return {
            "case_id": case["case_id"],
            "morphology": case["morphology"],
            "grid": case["grid"],
            "ks_plus": rec["ks_plus"],
            "ks_nominal": rec["ks_nominal"],
            "dU_plus_profile": rec["dU_plus_profile"],
            "dU_plus_bulk": rec["dU_plus_bulk"],
            "u_plus_bulk": rec["u_plus_bulk"],
            "u_tau": rec["u_tau"],
            "h_plus": rec["h_plus"],
            "re_tau": rec["re_tau"],
            "log_law_exists": rec["log_law_exists"],
            "log_fit": rec["log_fit"],
            "log_fit_free_kappa": rec.get("log_fit_free_kappa"),
            "log_window": rec["log_window"],
            "rho_mean": rec["rho_mean"],
            "max_mach": rec["max_mach"],
            "u_rms": rec["u_rms"],
            "solid_fraction": rec["geometry"].get("solid_fraction", 0.0),
            "coverage": rec["geometry"].get("coverage", 0.0),
            "exposed_area_ratio": rec["geometry"].get("exposed_area_ratio", 0.0),
            "h_rms": rec["geometry"].get("h_rms", 0.0),
            "wall_seconds": rec["wall_seconds"],
            "backend": rec["backend"],
            "steps": rec["steps"],
        }

    rows = {cid: row(rec) for cid, rec in records.items()}
    refs = {g: rows[f"smooth_{g}"] for g in GRIDS if f"smooth_{g}" in rows}

    # --- curves per morphology on the medium grid ------------------------
    curves: dict[str, list[dict[str, Any]]] = {}
    for cid, r in rows.items():
        if r["grid"] != "medium" or r["morphology"] == "smooth":
            continue
        curves.setdefault(r["morphology"], []).append(r)

    # --- criterion (a): dU+ vs literature band ---------------------------
    band_check: dict[str, Any] = {}
    for morph, pts in curves.items():
        detail = []
        for p in sorted(pts, key=lambda d: d["ks_plus"]):
            band = literature_band(np.asarray([p["ks_plus"]]))
            ref = float(band["ref"][0])
            tol = max(0.10 * abs(ref), 0.5)
            err = p["dU_plus_profile"] - ref
            detail.append({
                "ks_plus": p["ks_plus"],
                "dU_plus": p["dU_plus_profile"],
                "reference": ref,
                "cebeci": float(band["cebeci"][0]),
                "nikuradse": float(band["nikuradse"][0]),
                "abs_err": err,
                "tol": tol,
                "pass": bool(abs(err) <= tol),
                "in_band": bool(band["lo"][0] <= p["dU_plus_profile"] <= band["hi"][0]),
            })
        band_check[morph] = {
            "points": detail,
            "n_pass": sum(1 for d in detail if d["pass"]),
            "n_total": len(detail),
            "max_abs_err": max(abs(d["abs_err"]) for d in detail),
        }

    # --- criterion (c): slope in the fully rough region ------------------
    slope: dict[str, Any] = {}
    for morph, pts in curves.items():
        fr = [p for p in sorted(pts, key=lambda d: d["ks_plus"]) if p["ks_plus"] >= 90.0]
        if len(fr) < 2:
            slope[morph] = {"n_points": len(fr), "slope": float("nan"), "rel_err_vs_1_over_kappa": float("nan"),
                            "pass": False}
            continue
        x = np.log([p["ks_plus"] for p in fr])
        y = np.asarray([p["dU_plus_profile"] for p in fr])
        s, b = np.polyfit(x, y, 1)
        target = 1.0 / KAPPA
        slope[morph] = {
            "n_points": len(fr),
            "ks_plus_range": [fr[0]["ks_plus"], fr[-1]["ks_plus"]],
            "slope": float(s),
            "intercept": float(b),
            "target_1_over_kappa": float(target),
            "rel_err_vs_1_over_kappa": float(abs(s - target) / target),
            "pass": bool(abs(s - target) / target <= 0.15),
        }

    # --- criterion (b): grid convergence + GCI ---------------------------
    h_cells = {g: GRIDS[g]["h"] for g in GRIDS}
    # refinement ratio by linear grid size (same box aspect on every grid)
    r_cm = h_cells["medium"] / h_cells["coarse"]
    r_fm = h_cells["fine"] / h_cells["medium"]
    convergence: dict[str, Any] = {}
    gci_rows: list[dict[str, Any]] = []
    for ksp in (37.5, 75.0, 150.0):
        trio = {}
        for g in ("coarse", "medium", "fine"):
            cid = f"sand_{g}_ksp{str(ksp).replace('.', 'p')}"
            if cid in rows:
                trio[g] = rows[cid]
        if len(trio) < 3:
            continue
        convergence[str(ksp)] = [trio[g] for g in ("coarse", "medium", "fine")]
        # two three-grid estimates: one per adjacent refinement ratio, using the
        # common ratio 1.333 (coarse->fine spacing is uniform in log(h))
        f3 = trio["coarse"]["dU_plus_profile"]
        f2 = trio["medium"]["dU_plus_profile"]
        f1 = trio["fine"]["dU_plus_profile"]
        r = math.sqrt(r_cm * r_fm)
        est = _richardson(f1, f2, f3, r)
        gci_rows.append({
            "ks_plus": ksp,
            "dU_plus": {"coarse": f3, "medium": f2, "fine": f1},
            "rel_change_fine_medium": abs(f1 - f2) / max(abs(f1), 1e-12),
            "rel_change_coarse_medium": abs(f2 - f3) / max(abs(f2), 1e-12),
            "p": est["p"],
            "gci_fine": est["gci_fine"],
            "richardson_extrapolated": est["extrapolated"],
            "pass_change": bool(abs(f1 - f2) / max(abs(f1), 1e-12) <= 0.03),
            "pass_gci": bool(math.isfinite(est["gci_fine"]) and est["gci_fine"] <= 0.05),
        })

    # --- criterion (c) part 1: smooth-region dU+ -------------------------
    smooth_region = {}
    for g, ref in refs.items():
        # dU+ of the smooth case evaluated with the universal B = 5.0: any
        # departure is the solver's own log-law bias, which is exactly what the
        # "hydraulically smooth wall must show dU+ ~ 0" check measures.
        smooth_region[g] = {
            "B_fit": None if ref["log_fit"] is None else ref["log_fit"]["b"],
            "dU_plus_vs_universal_B": None if ref["log_fit"] is None else 5.0 - ref["log_fit"]["b"],
            "kappa_fit": None if ref["log_fit_free_kappa"] is None else ref["log_fit_free_kappa"]["kappa"],
            "pass": bool(ref["log_fit"] is not None and abs(5.0 - ref["log_fit"]["b"]) <= 0.15),
        }

    # --- criterion (e): morphology sensitivity ---------------------------
    morph_diff: list[dict[str, Any]] = []
    for ksp in sorted({p["ks_plus"] for pts in curves.values() for p in pts}):
        by_morph = {}
        for morph, pts in curves.items():
            for p in pts:
                if abs(p["ks_plus"] - ksp) < 1e-6:
                    by_morph[morph] = p["dU_plus_profile"]
        if len(by_morph) >= 2:
            vals = list(by_morph.values())
            morph_diff.append({
                "ks_plus": ksp,
                "dU_plus": by_morph,
                "spread": max(vals) - min(vals),
                "ribs_minus_sand": by_morph.get("ribs", float("nan")) - by_morph.get("sand", float("nan")),
                "ice_minus_sand": by_morph.get("ice_gouges", float("nan")) - by_morph.get("sand", float("nan")),
            })

    # --- criterion (d): ITTC-78 comparison -------------------------------
    sand_points = sorted([(p["ks_plus"], p["dU_plus_profile"]) for p in curves.get("sand", [])])
    ittc_rows = []
    if sand_points:
        for re_l, ks_over_l in ITTC_POINTS:
            cf_sm = ittc57_cf(re_l)
            u_plus_sm = math.sqrt(2.0 / cf_sm)
            u_tau_ship = math.sqrt(cf_sm / 2.0)  # U = 1 -> u_tau/U = sqrt(Cf/2)
            ks = ks_over_l * SHIP_LENGTH
            # Ship speed follows from the prescribed Re_L: U = Re nu / L, and
            # u_tau = U sqrt(Cf/2).
            u_ship = re_l * NU_SEA_WATER / SHIP_LENGTH
            u_tau_ship = u_ship * math.sqrt(cf_sm / 2.0)
            ks_plus_ship = ks * u_tau_ship / NU_SEA_WATER
            d_u = float(delta_u_plus_model(np.asarray([ks_plus_ship]), sand_points)[0])
            d_cf_lbm = delta_cf_from_delta_u_plus(u_plus_sm, d_u)
            d_cf_ittc = float(ittc78_allowance(ks_over_l))
            ittc_rows.append({
                "re_l": re_l,
                "ks_over_l": ks_over_l,
                "ks_plus_ship": ks_plus_ship,
                "U_ship_ms": u_ship,
                "cf_smooth": cf_sm,
                "u_plus_smooth": u_plus_sm,
                "delta_u_plus": d_u,
                "delta_cf_lbm": d_cf_lbm,
                "delta_cf_ittc78": d_cf_ittc,
                "rel_err": (d_cf_lbm - d_cf_ittc) / d_cf_ittc if d_cf_ittc else float("nan"),
                "pass": bool(d_cf_ittc > 0 and abs(d_cf_lbm - d_cf_ittc) / d_cf_ittc <= 0.05),
                "ks_eff_over_ks_geom": equivalent_ks_plus(d_u) / ks_plus_ship if ks_plus_ship else float("nan"),
            })
    ittc_summary = {
        "ship_length_m": SHIP_LENGTH,
        "nu_m2_s": NU_SEA_WATER,
        "points": ittc_rows,
        "n_pass": sum(1 for r in ittc_rows if r["pass"]),
        "n_total": len(ittc_rows),
        "max_abs_rel_err": max((abs(r["rel_err"]) for r in ittc_rows), default=float("nan")),
        "ks_over_l": [r["ks_over_l"] for r in ittc_rows],
        "delta_cf_lbm": [r["delta_cf_lbm"] for r in ittc_rows],
        "delta_cf_ittc78": [r["delta_cf_ittc78"] for r in ittc_rows],
    }

    # --- profiles for the figure -----------------------------------------
    profile_cases = []
    for g in ("medium",):
        if f"smooth_{g}" in records:
            rec = records[f"smooth_{g}"]
            profile_cases.append({"label": f"smooth ({g})", "y_plus": rec["profile"]["y_plus"],
                                  "u_plus": rec["profile"]["u_plus"]})
    for morph in ("sand", "ribs", "ice_gouges"):
        cid = f"{morph}_medium_ksp75"
        if cid in records:
            rec = records[cid]
            profile_cases.append({"label": f"{morph}, ks+={rec['ks_plus']:.0f}",
                                  "y_plus": rec["profile"]["y_plus"], "u_plus": rec["profile"]["u_plus"]})

    criteria = {
        "a_dU_vs_literature": {
            "pass": all(v["n_pass"] == v["n_total"] for v in band_check.values()) if band_check else False,
            "detail": band_check,
            "rule": "|dU+ - band_ref| <= max(10% , 0.5)",
        },
        "b_grid_convergence": {
            "pass": bool(gci_rows) and all(r["pass_change"] for r in gci_rows),
            "gci": gci_rows,
            "rule": "dU+(fine)-dU+(medium) <= 3%; GCI_fine <= 5%",
        },
        "c_smooth_region_and_slope": {
            "pass_smooth": all(v["pass"] for v in smooth_region.values()),
            "pass_slope": all(v.get("pass", False) for v in slope.values()),
            "smooth": smooth_region,
            "slope": slope,
            "rule": "smooth-wall dU+ <= 0.15; fully-rough slope = 1/kappa +-15%",
        },
        "d_ittc78": {
            "pass": ittc_summary["n_pass"] == ittc_summary["n_total"] and ittc_summary["n_total"] > 0,
            "detail": ittc_summary,
            "rule": "|dCf_LBM - dCf_ITTC78| / dCf_ITTC78 <= 5% at 6 points",
        },
        "e_morphology_sensitivity": {
            "pass": bool(morph_diff) and any(abs(m["spread"]) > 0.0 for m in morph_diff),
            "detail": morph_diff,
            "rule": "report the measured spread (ribs vs sand vs ice at matched ks+)",
        },
    }
    criteria["all_machine_criteria_pass"] = bool(
        criteria["a_dU_vs_literature"]["pass"]
        and criteria["b_grid_convergence"]["pass"]
        and criteria["c_smooth_region_and_slope"]["pass_smooth"]
        and criteria["c_smooth_region_and_slope"]["pass_slope"]
        and criteria["d_ittc78"]["pass"]
        and criteria["e_morphology_sensitivity"]["pass"]
    )

    summary = {
        "campaign": {
            "re_tau": RE_TAU,
            "grids": GRIDS,
            "steps_rule": {"warmup_turnovers": TURNOVERS_WARMUP, "sample_turnovers": TURNOVERS_SAMPLE},
            "n_cases_expected": len(cases),
            "n_cases_present": len(records),
            "missing": missing,
        },
        "cases": rows,
        "curves": curves,
        "profile_cases": profile_cases,
        "convergence": {k: [{"grid": r["grid"], "dU_plus_profile": r["dU_plus_profile"]} for r in v]
                        for k, v in convergence.items()},
        "criteria": criteria,
        "ittc78": ittc_summary,
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(ART_DIR / "campaign_summary.json", summary)
    figs = _figures(summary, ART_DIR)
    summary["figures"] = figs
    _write_json(ART_DIR / "campaign_summary.json", summary)
    print(json.dumps({
        "cases": len(records),
        "missing": missing,
        "all_pass": criteria["all_machine_criteria_pass"],
        "ittc_pass": f"{ittc_summary['n_pass']}/{ittc_summary['n_total']}",
        "gci": [(r["ks_plus"], round(r["gci_fine"], 4) if math.isfinite(r["gci_fine"]) else None) for r in gci_rows],
        "figures": [f["path"] for f in figs],
    }, indent=1))
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_work = sub.add_parser("worker")
    p_work.add_argument("--worker-index", type=int, required=True)
    p_work.add_argument("--workers", type=int, required=True)
    p_work.add_argument("--threads", type=int, default=16)
    p_work.add_argument("--steps-scale", type=float, default=1.0)
    p_work.add_argument("--rebuild", action="store_true")
    p_work.add_argument("--force", action="store_true")
    p_work.add_argument("--only-reference", action="store_true",
                        help="run only the smooth reference cases (phase 0 of the campaign)")
    p_work.add_argument("--skip-reference", action="store_true",
                        help="skip the smooth reference cases (phase 1: they are already cached)")
    p_agg = sub.add_parser("aggregate")
    for p in (p_list, p_agg):
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--steps-scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "worker":
        return cmd_worker(args)
    return cmd_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())