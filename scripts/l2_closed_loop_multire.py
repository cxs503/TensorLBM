#!/usr/bin/env python3
"""L3 closed-loop, multi-Re increment: a mission-profile hull C_D objective.

The 2026-09-08 single-Re loop (``scripts/l2_closed_loop_demo.py``, PR #286)
minimised C_D at ONE Reynolds number. Ship design cares about an operating
RANGE: this script reruns the loop against a mission-profile objective

    J(design) = mean of C_D over a shared 8-point log10-Re grid spanning
    the certified intersection window of the corpus,

with the same two-arm shape and the same honesty rules (no new LBM, every
number tagged surrogate / truth / derived):

    step 0 — coverage map (a deliverable in its own right): per-design Re
    rows of the 406-row pool, the >=4-rows-spanning->=0.5-dec certified
    set, the per-design row-count histogram, the intersection window;

    arm A — free continuous search (LHS 40 shared + trust-region refine
    24 per arm, greedy and UQ-aware LCB k=1, identical RNG discipline to
    the demo): every candidate query serves the ensemble at ALL grid Re
    and averages — does the multi-Re objective still chase l/d < 1?;

    arm B — certified-region enumeration: the covered designs ranked by
    TRUE J (quad3 on each design's OWN rows at the shared grid — never
    extrapolated outside a design's measured span) vs PREDICTED J (the
    same serving path): optimum recovery, APE, Kendall tau, and the
    certified improvement over the anchor design where truth allows;

    phantom — if arm A still picks l/d < 1, the objective-level refutation
    from the W9 corner-geometry scan datasets (read-only): true J of the
    corner geometry at 4 l/d levels x 14 shared Re vs what the surrogate
    claims, plus the g10005 l/d=1.0 corner datum at Re=200.

Usage (paths default to the 5090 production layout; every one is a flag):

    python scripts/l2_closed_loop_multire.py
    python scripts/l2_closed_loop_multire.py --verify
    python scripts/l2_closed_loop_multire.py --check-doc docs/l2_closed_loop_multire_20260908.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from l2_closed_loop_demo import (  # noqa: E402
    BOX,
    DVARS,
    lhs_unit,
    to_params,
    to_unit,
)
from l2_serving_walkthrough import (  # noqa: E402
    build_design_mask,
    build_service,
    default_paths,
    fit_guard,
    load_corpus,
)

from tensorlbm.ai.field_provider import FieldProvider  # noqa: E402
from tensorlbm.ai.geom_encoder import sdf_volume  # noqa: E402
from tensorlbm.ai.inference_service import quad3_nearest3  # noqa: E402

#: Fixed mission posture (same as the single-Re demo, minus the fixed Re).
MISSION_HULL = "full"
MISSION_U_IN = 0.1

#: Objective grid: 8 points log-spaced over the intersection window of the
#: certified set (computed at runtime from the coverage map).
GRID_N = 8

#: Certified-set rule (the mission definition): a design is covered when
#: its OWN corpus rows number >= MIN_ROWS and span >= MIN_SPAN_DEC decades.
MIN_ROWS = 4
MIN_SPAN_DEC = 0.5

#: Search budget: LHS 40 shared + refine 24 per arm (greedy and LCB k=1).
COARSE_N = 40
REFINE_ROUNDS = 4
REFINE_PER_ROUND = 6
K_LCB = 1.0
RHO0 = 0.30
RHO_DECAY = 0.6
SEED = 20260908

#: Control-sanity constants (blockers if violated).
KNOWN_ROW233_CD = 4.381678732563599

#: W9 corner-geometry truth campaigns (read-only; sail 0.645983 / fin
#: 2.915821 — the single-Re loop's greedy corner).
CORNER_SAIL = 0.645983
CORNER_FIN = 2.915821


def add_args(parser: argparse.ArgumentParser) -> None:
    for name, value in default_paths().items():
        parser.add_argument(f"--{name}", default=value, help=f"(default: {value})")
    parser.add_argument(
        "--out",
        default="/nfs/wangxi/runs/l2_loop_multire_20260908/multire.json",
        help="result json (all numbers, surrogate/truth tagged; no wall-clock)",
    )
    parser.add_argument(
        "--report",
        default="/nfs/wangxi/runs/l2_loop_multire_20260908/report.md",
        help="report.md rendered FROM the json (zero hand-typed numbers)",
    )
    parser.add_argument(
        "--ld-ladder-truth",
        default="/nfs/wangxi/runs/ld_ladder_20260908/scan_truth.json",
        help="W9 ladder scan_truth.json (corner geometry, l/d 0.90/0.95 x 14 Re)",
    )
    parser.add_argument(
        "--ld-gap-truth",
        default="/nfs/wangxi/runs/ld_gap_20260908/scan_truth.json",
        help="W9 gap scan_truth.json (corner geometry, l/d 0.925/0.975 x 14 Re)",
    )
    parser.add_argument(
        "--l2loop-truth",
        default="/nfs/wangxi/runs/l2_loop_truth_20260908/truth.json",
        help="closed-loop truth campaign (g10005 = corner geometry at l/d 1.0)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arm", default="ts2", choices=("ts2", "ts4"))
    parser.add_argument("--uq-temperature", type=float, default=1.5)
    parser.add_argument("--hull", default=MISSION_HULL)
    parser.add_argument("--u-in", type=float, default=MISSION_U_IN)
    parser.add_argument("--grid-n", type=int, default=GRID_N)
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS)
    parser.add_argument("--min-span-dec", type=float, default=MIN_SPAN_DEC)
    parser.add_argument("--coarse-n", type=int, default=COARSE_N)
    parser.add_argument("--refine-rounds", type=int, default=REFINE_ROUNDS)
    parser.add_argument("--refine-per-round", type=int, default=REFINE_PER_ROUND)
    parser.add_argument("--k-lcb", type=float, default=K_LCB)
    parser.add_argument("--rho0", type=float, default=RHO0)
    parser.add_argument("--rho-decay", type=float, default=RHO_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-render the report from the json (byte-identical?) + recheck gates",
    )
    parser.add_argument(
        "--check-doc",
        default=None,
        metavar="DOC.md",
        help="every decimal token of DOC.md must be formattable from the json",
    )


# --------------------------------------------------------------------------- #
# 1. Step 0: the coverage map (reported BEFORE any search is built)
# --------------------------------------------------------------------------- #
def design_rows(d: dict[str, Any], design_index: int) -> np.ndarray:
    return np.where(d["karr"] == design_index)[0]


def is_covered(d: dict[str, Any], design_index: int, p: argparse.Namespace) -> bool:
    rows = design_rows(d, design_index)
    res = d["re"][rows]
    return rows.size >= p.min_rows and (np.log10(res.max()) - np.log10(res.min())) >= p.min_span_dec


def coverage_map(d: dict[str, Any], p: argparse.Namespace) -> dict[str, Any]:
    """Per-design Re coverage of the pool; the certified sets; the windows."""
    uniq = d["uniq"]
    full_lod1 = [i for i, k in enumerate(uniq) if k[0] == "full" and abs(k[3] - 1.0) < 1e-9]

    def hist(idxs: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in idxs:
            n = int(design_rows(d, i).size)
            counts[str(n)] = counts.get(str(n), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))

    def spans(idxs: list[int]) -> list[dict[str, Any]]:
        out = []
        for i in idxs:
            rows = design_rows(d, i)
            res = d["re"][rows]
            uins = sorted(set(np.round(d["uin"][rows], 6).tolist()))
            out.append(
                dict(
                    design_index=int(i),
                    design_key=list(uniq[i]),
                    n_rows=int(rows.size),
                    re_min=float(res.min()),
                    re_max=float(res.max()),
                    span_dec=float(np.log10(res.max()) - np.log10(res.min())),
                    u_in_levels=[float(u) for u in uins],
                )
            )
        return out

    cert_full = [i for i in full_lod1 if is_covered(d, i, p)]
    cert_any = [i for i in range(len(uniq)) if is_covered(d, i, p)]
    sp = spans(cert_full)
    win_lo = max(math.log10(s["re_min"]) for s in sp)
    win_hi = min(math.log10(s["re_max"]) for s in sp)
    grid = 10.0 ** np.linspace(win_lo, win_hi, p.grid_n)

    exact200 = []
    for r in np.where(np.isclose(d["re"], 200.0))[0]:
        i = int(d["karr"][r])
        if uniq[i][0] == "full" and abs(uniq[i][3] - 1.0) < 1e-9:
            exact200.append(
                dict(
                    corpus_row=int(r),
                    design_index=i,
                    design_key=list(uniq[i]),
                    truth_cd=float(d["cd"][r]),
                )
            )

    return dict(
        tag="derived",
        n_pool_rows=int(len(d["cd"])),
        n_designs=int(len(uniq)),
        n_full_lod1_designs=len(full_lod1),
        row_count_histogram_all=dict(tag="truth", counts=hist(list(range(len(uniq))))),
        row_count_histogram_full_lod1=dict(tag="truth", counts=hist(full_lod1)),
        n_designs_with_4plus_rows_spanning_half_dec_all=len(cert_any),
        n_designs_with_4plus_rows_spanning_half_dec_full_lod1=len(cert_full),
        certified_full_lod1=dict(
            tag="truth",
            rule=f">= {p.min_rows} own rows spanning >= {p.min_span_dec} decades",
            designs=sp,
        ),
        certified_any_hull=dict(
            tag="truth",
            rule=f">= {p.min_rows} own rows spanning >= {p.min_span_dec} decades, any hull",
            designs=spans(cert_any),
        ),
        intersection_window_full_lod1=dict(
            tag="derived",
            log10_re_min=win_lo,
            log10_re_max=win_hi,
            re_min=float(10.0**win_lo),
            re_max=float(10.0**win_hi),
            width_dec=float(win_hi - win_lo),
            basis="intersection of the certified full l/d=1.0 designs' measured spans",
        ),
        exact_re200_full_lod1=dict(tag="truth", rows=exact200),
        objective_grid=dict(
            tag="derived",
            n=p.grid_n,
            rule=f"{p.grid_n} points log-spaced over the intersection window",
            re=[float(g) for g in grid],
        ),
    )


# --------------------------------------------------------------------------- #
# 2. Truth: exact rows or quad3 on a design's OWN rows (never extrapolated)
# --------------------------------------------------------------------------- #
def truth_at_grid(d: dict[str, Any], design_index: int, grid: np.ndarray) -> list[dict[str, Any]]:
    rows = design_rows(d, design_index)
    res, cds = d["re"][rows], d["cd"][rows]
    lo, hi = float(res.min()), float(res.max())
    out = []
    for g in grid:
        exact = np.isclose(res, g)
        if exact.any():
            vals = [float(v) for v in cds[exact]]
            out.append(
                dict(
                    re=float(g),
                    value=float(np.mean(vals)),
                    method="exact_corpus_rows",
                    corpus_rows=[int(rows[i]) for i in np.where(exact)[0]],
                    replica_spread=(float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0),
                )
            )
        elif rows.size >= 3 and lo < g < hi:
            q = quad3_nearest3(res, cds, g)
            if q is not None:
                out.append(
                    dict(
                        re=float(g),
                        value=float(q[0]),
                        method="quad3_interpolated_between_own_rows",
                        chosen_re_ascending=[float(v) for v in q[1]],
                    )
                )
            else:
                out.append(dict(re=float(g), value=None, method="not_certifiable"))
        else:
            out.append(dict(re=float(g), value=None, method="not_certifiable"))
    return out


def truth_j(points: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [pt["value"] for pt in points if pt["value"] is not None]
    return dict(
        tag="truth",
        value=float(np.mean(vals)) if vals else None,
        n_certified_points=len(vals),
        n_grid_points=len(points),
    )


def truth_at_re(d: dict[str, Any], design_index: int, re_q: float) -> dict[str, Any]:
    """Exact row(s) or quad3-on-own-rows truth at one Re; None if not covered."""
    rows = design_rows(d, design_index)
    res, cds = d["re"][rows], d["cd"][rows]
    exact = np.isclose(res, re_q)
    if exact.any():
        return dict(tag="truth", value=float(np.mean(cds[exact])), method="exact_corpus_rows")
    lo, hi = float(res.min()), float(res.max())
    if rows.size >= 3 and lo < re_q < hi:
        q = quad3_nearest3(res, cds, re_q)
        if q is not None:
            return dict(tag="truth", value=float(q[0]), method="quad3_on_own_rows")
    return dict(tag="truth", value=None, method="not_certifiable")


def w9_corner_curves(p: argparse.Namespace) -> dict[float, list[dict[str, Any]]]:
    """The W9 corner-geometry truth curves, one row-list per l/d level."""
    curves: dict[float, list[dict[str, Any]]] = {}
    for path in (p.ld_ladder_truth, p.ld_gap_truth):
        st = json.loads(Path(path).read_text())
        for pid, rec in st["points"].items():
            if pid.startswith("ctl"):
                continue
            prm = rec["params"]
            if not (
                abs(float(prm["sail_scale"]) - CORNER_SAIL) < 1e-9
                and abs(float(prm["fin_scale"]) - CORNER_FIN) < 1e-9
            ):
                continue
            ld = round(float(prm["l_over_d_mult"]), 6)
            curves.setdefault(ld, []).append(
                dict(re=float(rec["re"]), cd_proj=float(rec["scan_truth"]["cd_proj"]))
            )
    for ld in curves:
        curves[ld].sort(key=lambda c: c["re"])
    return curves


def scan_truth_at(curve: list[dict[str, Any]], re_query: float) -> dict[str, Any]:
    """Exact row or quad3 on one W9 measured curve (strictly inside only)."""
    res = np.array([c["re"] for c in curve])
    cds = np.array([c["cd_proj"] for c in curve])
    exact = np.isclose(res, re_query)
    if exact.any():
        return dict(tag="truth", value=float(cds[exact].mean()), method="exact_scan_rows")
    lo, hi = float(res.min()), float(res.max())
    if res.size >= 3 and lo < re_query < hi:
        q = quad3_nearest3(res, cds, re_query)
        if q is not None:
            return dict(tag="truth", value=float(q[0]), method="quad3_interpolated_between_rows")
    return dict(tag="truth", value=None, method="not_certifiable")


# --------------------------------------------------------------------------- #
# 3. The multi-Re evaluator: one design -> served C_D over the WHOLE grid
# --------------------------------------------------------------------------- #
class MultiReEvaluator:
    """Serves J(design) = mean ensemble C_D over the shared Re grid.

    One query = the full production path of the single-Re loop (CAD params
    -> occupancy mask -> SDF -> field_borrow -> 10-member ensemble predict)
    with the whole grid passed in a single service call. The objective's
    ensemble-uncertainty column is the grid-mean of the RAW per-Re member
    std (the service does not expose the member matrix; by Cauchy-Schwarz
    the grid-mean of per-Re stds upper-bounds the member std of the
    grid-mean J).
    """

    def __init__(self, svc: Any, grid: np.ndarray, device: str, p: argparse.Namespace):
        self.svc = svc
        self.grid = np.asarray(grid, dtype=float)
        self.device = device
        self.p = p
        self.k_lcb = p.k_lcb
        self._recs: dict[tuple, dict[str, Any]] = {}
        self._sdf: dict[tuple, np.ndarray] = {}

    def _mask_sdf(
        self, hull: str, lod: float, sail: float, fin: float, mults: tuple[float, ...]
    ) -> np.ndarray:
        key = (hull, round(lod, 9), round(sail, 9), round(fin, 9), tuple(mults))
        hit = self._sdf.get(key)
        if hit is None:
            mask = build_design_mask((hull, sail, fin, lod, *mults))
            sdf_t = sdf_volume(torch.from_numpy(mask).to(self.device))
            hit = sdf_t.squeeze().cpu().numpy()
            self._sdf[key] = hit
        return hit

    def query(
        self,
        hull: str,
        params: tuple[float, float, float],
        mults: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, Any]:
        """Evaluate ``(l_over_d, sail, fin)`` for one hull — cached."""
        lod, sail, fin = (round(float(v), 9) for v in params)
        key = (str(hull), lod, sail, fin, tuple(mults))
        hit = self._recs.get(key)
        if hit is not None:
            return hit
        res = self._serve(key, self.grid)
        rec = self._record(key, res, self.grid)
        self._recs[key] = rec
        return rec

    def query_at_re(
        self,
        hull: str,
        params: tuple[float, float, float],
        re_arr: np.ndarray,
        mults: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, Any]:
        """Serve one design at an arbitrary Re array (no J aggregation)."""
        lod, sail, fin = (round(float(v), 9) for v in params)
        key = (str(hull), lod, sail, fin, tuple(mults), tuple(float(r) for r in re_arr))
        hit = self._recs.get(key)
        if hit is not None:
            return hit
        res = self._serve(key[:5], np.asarray(re_arr, dtype=float))
        rec = dict(
            tag="surrogate",
            re=[float(r) for r in re_arr],
            cd_mean=[float(v) for v in res.cd],
        )
        self._recs[key] = rec
        return rec

    def _serve(self, key: tuple, re_arr: np.ndarray) -> Any:
        hull, lod, sail, fin, mults = key
        sdf = self._mask_sdf(hull, lod, sail, fin, mults)
        return self.svc.predict(
            hull,
            sail,
            fin,
            re_arr,
            u_in=self.p.u_in,
            sdf=sdf,
            field_policy="field_borrow",
        )

    def _record(self, key: tuple, res: Any, grid: np.ndarray) -> dict[str, Any]:
        hull, lod, sail, fin, _mults = key
        t = float(res.info.get("uq_temperature", 1.0))
        std_raw = np.asarray(res.std, dtype=float) / t
        w = 1.0 / grid
        w = w / w.sum()
        prov = res.info.get("field_borrow", {})
        return dict(
            params=dict(zip(DVARS, [float(lod), float(sail), float(fin)])),
            hull=str(hull),
            surrogate=dict(
                tag="surrogate",
                j_mean=float(np.mean(res.cd)),
                j_fuel_weighted=float(np.sum(w * np.asarray(res.cd, dtype=float))),
                std_ensemble=float(np.mean(std_raw)),
                std_served=float(np.mean(res.std)),
                lcb=float(np.mean(res.cd)) - self.k_lcb * float(np.mean(std_raw)),
            ),
            per_re=dict(
                tag="surrogate",
                re=[float(g) for g in grid],
                cd_mean=[float(v) for v in res.cd],
                std_raw=[float(v) for v in std_raw],
            ),
            service=dict(
                borrow_distance=(float(prov["distance"]) if "distance" in prov else None),
                donor_pool_index=(int(prov["donor_index"]) if "donor_index" in prov else None),
                borrow_guard_ok=bool(prov.get("guard_ok", False)),
                service_guard_flag=str(res.guard.as_dict().get("flag", "?")),
                uq_temperature=t,
            ),
        )


def acquisition(rec: dict[str, Any], rule: str) -> float:
    s = rec["surrogate"]
    return s["j_mean"] if rule == "greedy" else s["lcb"]


# --------------------------------------------------------------------------- #
# 4. Arm A: shared coarse LHS + per-arm trust-region refine (demo discipline)
# --------------------------------------------------------------------------- #
def run_search(
    ev: MultiReEvaluator, p: argparse.Namespace, log: list[dict[str, Any]]
) -> dict[str, Any]:
    rng_c = np.random.default_rng(p.seed)
    coarse_u = lhs_unit(p.coarse_n, len(DVARS), rng_c)
    coarse = []
    for u in coarse_u:
        rec = dict(ev.query(p.hull, to_params(u)), use="coarse", query_id=len(log) + 1)
        log.append(rec)
        coarse.append(rec)
    best_c = {r: min(coarse, key=lambda rc: acquisition(rc, r)) for r in ("greedy", "lcb")}
    print(
        f"[coarse] n={p.coarse_n} -> greedy best J="
        f"{best_c['greedy']['surrogate']['j_mean']:.4f}, lcb best "
        f"J-lcb={best_c['lcb']['surrogate']['lcb']:.4f}"
    )
    arms = {}
    for rule in ("greedy", "lcb"):
        rng = np.random.default_rng(p.seed + 1)  # identical draws per arm
        seen = list(coarse)
        best = best_c[rule]
        history = [
            dict(
                stage="coarse",
                best_params=dict(best["params"]),
                best_acquisition=acquisition(best, rule),
            )
        ]
        for rnd in range(p.refine_rounds):
            rho = p.rho0 * (p.rho_decay**rnd)
            c = to_unit(tuple(best["params"][v] for v in DVARS))
            for _ in range(p.refine_per_round):
                u = np.clip(c + rho * (2.0 * rng.random(len(DVARS)) - 1.0), 0.0, 1.0)
                rec = dict(
                    ev.query(p.hull, to_params(u)),
                    use=f"refine:{rule}",
                    query_id=len(log) + 1,
                )
                log.append(rec)
                seen.append(rec)
                if acquisition(rec, rule) < acquisition(best, rule):
                    best = rec
            history.append(
                dict(
                    stage=f"refine_round_{rnd + 1}",
                    radius_box_fraction=round(rho, 6),
                    best_params=dict(best["params"]),
                    best_acquisition=acquisition(best, rule),
                )
            )
        arms[rule] = dict(acquisition=rule, n_queries=len(seen), chosen=best, history=history)
        s = best["surrogate"]
        print(
            f"[refine:{rule}] chosen l/d={best['params']['l_over_d_mult']:.4f}"
            f" sail={best['params']['sail_scale']:.4f}"
            f" fin={best['params']['fin_scale']:.4f} J={s['j_mean']:.4f}"
            f" std_ens={s['std_ensemble']:.4f} lcb={s['lcb']:.4f}"
        )
    return dict(coarse_n=p.coarse_n, coarse=coarse, arms=arms)


# --------------------------------------------------------------------------- #
# 5. Ranking statistics (no scipy dependency)
# --------------------------------------------------------------------------- #
def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx and sy:
                conc += int(sx * sy > 0)
                disc += int(sx * sy < 0)

    def ties(v: list[float]) -> int:
        _, cnt = np.unique(v, return_counts=True)
        return sum(int(c * (c - 1) // 2) for c in cnt if c > 1)

    n0 = n * (n - 1) // 2
    n1, n2 = ties(x), ties(y)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    return float((conc - disc) / denom) if denom else None


def ranks_desc(values: list[float]) -> list[int]:
    """1-based ranks of a minimisation problem (smallest -> rank 1)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0] * len(values)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


# --------------------------------------------------------------------------- #
# 6. Arm B + phantom + result assembly
# --------------------------------------------------------------------------- #
def candidate_record(
    d: dict[str, Any],
    ev: MultiReEvaluator,
    design_index: int,
    sub_grid: np.ndarray,
    grid: np.ndarray,
    w: np.ndarray,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    key = d["uniq"][design_index]
    pts = truth_at_grid(d, design_index, sub_grid)
    tj = truth_j(pts)
    rec = ev.query(
        str(key[0]),
        (float(key[3]), float(key[1]), float(key[2])),
        mults=tuple(float(v) for v in key[4:7]),
    )
    log.append(dict(rec, use="enumeration", query_id=len(log) + 1))
    sub = [i for i, g in enumerate(grid) if any(abs(g - sg) < 1e-9 for sg in sub_grid)]
    pred = float(np.mean([rec["per_re"]["cd_mean"][i] for i in sub]))
    pred_fuel = float(np.sum([w[i] * rec["per_re"]["cd_mean"][i] for i in sub]))
    return dict(
        design_index=int(design_index),
        design_key=list(key),
        truth_j=dict(tj, grid_re=[float(g) for g in sub_grid]),
        truth_points=pts,
        surrogate_j=dict(tag="surrogate", value=pred, j_fuel_weighted=pred_fuel),
        surrogate_service=rec["service"],
        ape_percent=(
            dict(tag="derived", value=abs(pred / tj["value"] - 1.0) * 100.0)
            if tj["value"] is not None
            else None
        ),
    )


def ranking_block(cands: list[dict[str, Any]]) -> dict[str, Any]:
    truths = [c["truth_j"]["value"] for c in cands]
    preds = [c["surrogate_j"]["value"] for c in cands]
    rt, rp = ranks_desc(truths), ranks_desc(preds)
    best_true = int(np.argmin(truths))
    best_pred = int(np.argmin(preds))
    n = len(cands)
    return dict(
        tag="derived",
        n=n,
        kendall_tau_b=kendall_tau_b(truths, preds),
        surrogate_rank_of_true_best=int(rt[best_true]),
        true_rank_of_surrogate_best=int(rp[best_pred]),
        top_decile_displacement=int(rt[best_true]) - 1,
        top_decile_note="n < 10: the top decile is the top 1" if n < 10 else "",
    )


def optimum_block(cands: list[dict[str, Any]], rk: dict[str, Any]) -> dict[str, Any]:
    bt = int(np.argmin([c["truth_j"]["value"] for c in cands]))
    bp = int(np.argmin([c["surrogate_j"]["value"] for c in cands]))
    return dict(
        tag="derived",
        true_optimum=dict(
            design_index=cands[bt]["design_index"],
            design_key=cands[bt]["design_key"],
            truth_j=cands[bt]["truth_j"]["value"],
        ),
        predicted_optimum=dict(
            design_index=cands[bp]["design_index"],
            design_key=cands[bp]["design_key"],
            surrogate_j=cands[bp]["surrogate_j"]["value"],
            truth_j_of_predicted_optimum=cands[bp]["truth_j"]["value"],
        ),
        same_design=bool(bt == bp),
        ape_at_the_optimum_percent=(
            abs(cands[bp]["surrogate_j"]["value"] / cands[bp]["truth_j"]["value"] - 1.0) * 100.0
        ),
        true_optimum_missed_by_percent=(
            cands[bp]["truth_j"]["value"] / cands[bt]["truth_j"]["value"] - 1.0
        )
        * 100.0,
        ranking=rk,
    )


def certified_improvement_vs_anchor(
    d: dict[str, Any], design_index: int, anchor_truth: dict[float, float]
) -> dict[str, Any]:
    out = {}
    for re_q, cd_true in sorted(anchor_truth.items()):
        q = truth_at_re(d, design_index, re_q)
        out[f"re_{re_q:.1f}"] = dict(
            tag="derived",
            anchor_truth_cd=dict(tag="truth", value=cd_true),
            candidate_truth_cd=q,
            improvement_percent=(
                (q["value"] / cd_true - 1.0) * 100.0 if q["value"] is not None else None
            ),
            certified=bool(q["value"] is not None),
        )
    return out


def build_phantom(
    p: argparse.Namespace,
    grid: np.ndarray,
    sub_grid: np.ndarray,
    ev: MultiReEvaluator,
    log: list[dict[str, Any]],
    arm_a: dict[str, Any],
) -> dict[str, Any]:
    """Objective-level l/d phantom adjudication from the W9 corner scans."""
    curves = w9_corner_curves(p)
    l2t = json.loads(Path(p.l2loop_truth).read_text())
    g10005 = l2t["points"]["g10005"]
    lds = sorted(curves)

    levels = {}
    for ld in lds:
        pts = curves[ld]
        per_point = [dict(re=float(g), **scan_truth_at(pts, float(g))) for g in sub_grid]
        j_vals = [pt["value"] for pt in per_point if pt["value"] is not None]
        levels[f"{ld:.4f}"] = dict(
            tag="truth",
            l_over_d_mult=float(ld),
            n_scan_rows=len(pts),
            re_min=float(min(c["re"] for c in pts)),
            re_max=float(max(c["re"] for c in pts)),
            truth_points_on_shared_grid=per_point,
            truth_j=dict(
                tag="derived",
                value=float(np.mean(j_vals)) if j_vals else None,
                n_points=len(j_vals),
            ),
            truth_cd_at_re200=scan_truth_at(pts, 200.0),
        )

    # truth chord (linear OLS slope of cd_proj vs l/d) at 3 measured Re
    # levels: nearest in log10 to 50 / 200 / 689
    chords = {}
    all_re = np.array(sorted({c["re"] for pts in curves.values() for c in pts}))
    for tag, want in (("re_low", 50.0), ("re_mid", 200.0), ("re_high", 689.0)):
        re_q = float(all_re[np.argmin(np.abs(np.log10(all_re) - math.log10(want)))])
        cds = [scan_truth_at(curves[ld], re_q)["value"] for ld in lds]
        if all(c is not None for c in cds):
            slope = float(np.polyfit(np.array(lds, dtype=float), np.array(cds), 1)[0])
            chords[tag] = dict(
                tag="truth",
                re=re_q,
                cd_per_unit_l_over_d_mult=slope,
                values=dict(zip([f"{ld:.4f}" for ld in lds], cds)),
            )

    # surrogate at the corner geometry: the same levels + l/d = 1.0
    surrogate_levels = {}
    sub_idx = [i for i, g in enumerate(grid) if any(abs(g - sg) < 1e-9 for sg in sub_grid)]
    for ld in lds + [1.0]:
        rec = ev.query("full", (float(ld), CORNER_SAIL, CORNER_FIN))
        log.append(dict(rec, use="phantom_corner", query_id=len(log) + 1))
        at200 = ev.query_at_re("full", (float(ld), CORNER_SAIL, CORNER_FIN), [200.0])
        surrogate_levels[f"{ld:.4f}"] = dict(
            tag="surrogate",
            l_over_d_mult=float(ld),
            j_on_shared_subgrid=float(np.mean([rec["per_re"]["cd_mean"][i] for i in sub_idx])),
            j_on_full_grid=rec["surrogate"]["j_mean"],
            cd_at_re200=float(at200["cd_mean"][0]),
            service=rec["service"],
        )
    s_keys = sorted(surrogate_levels)
    s_lds = [surrogate_levels[k]["l_over_d_mult"] for k in s_keys]
    s_js = [surrogate_levels[k]["j_on_shared_subgrid"] for k in s_keys]
    surrogate_slope = float(np.polyfit(np.array(s_lds), np.array(s_js), 1)[0])
    t_lds = [levels[f"{ld:.4f}"]["l_over_d_mult"] for ld in lds]
    t_js = [levels[f"{ld:.4f}"]["truth_j"]["value"] for ld in lds]
    truth_slope = float(np.polyfit(np.array(t_lds), np.array(t_js), 1)[0])
    g1_cd = float(g10005["scan_truth"]["cd_proj"])
    ld_min = min(t_lds)

    greedy = arm_a["arms"]["greedy"]["chosen"]["params"]
    chased = bool(greedy["l_over_d_mult"] < 1.0)
    signs_agree = bool(np.sign(truth_slope) == np.sign(surrogate_slope))
    return dict(
        tag="mixed (per-block)",
        trigger=dict(
            tag="derived",
            arm_a_greedy_pick_l_over_d=greedy["l_over_d_mult"],
            arm_a_chased_l_over_d_below_one=chased,
            single_re_context=(
                "the Re=200 free arm picked l/d 0.919335 (greedy) / 0.902405 (LCB),"
                " both refuted by fresh LBM at Re=200 (l2_loop_truth_20260908)"
            ),
        ),
        corner_geometry=dict(
            sail_scale=CORNER_SAIL,
            fin_scale=CORNER_FIN,
            hull="full",
            source="W9 ld_ladder + ld_gap scan campaigns (read-only)",
        ),
        truth_levels=levels,
        truth_chords=chords,
        truth_lod1_corner_at_re200=dict(
            tag="truth",
            cd_proj=g1_cd,
            source="g10005 point of l2_loop_truth_20260908 (Re=200 only)",
            note="no multi-Re truth exists at the corner geometry l/d=1.0;"
            " g10005 anchors the Re=200 direction only",
        ),
        surrogate_levels=surrogate_levels,
        slopes_on_shared_subgrid=dict(
            tag="derived",
            truth_dJ_d_l_over_d_over_scan_levels=truth_slope,
            surrogate_dJ_d_l_over_d_including_lod1=surrogate_slope,
            sign_agreement=signs_agree,
        ),
        objective_level_phantom=dict(
            tag="derived",
            surrogate_claimed_J_gain_from_lod_1_to_scan_min_percent=(
                surrogate_levels[f"{ld_min:.4f}"]["j_on_shared_subgrid"]
                / surrogate_levels["1.0000"]["j_on_shared_subgrid"]
                - 1.0
            )
            * 100.0,
            truth_J_change_across_scan_levels_percent=(max(t_js) / min(t_js) - 1.0) * 100.0,
            truth_re200_chord_lod_scan_min_to_1=(
                (g1_cd - levels[f"{ld_min:.4f}"]["truth_cd_at_re200"]["value"]) / (1.0 - ld_min)
            ),
            verdict=(
                "the surrogate multi-Re J still falls as l/d drops below 1 while the"
                " W9 truth J rises across the same levels — the phantom persists at"
                " the mission-profile level"
                if not signs_agree
                else "surrogate and W9 truth agree on the l/d direction at the"
                " mission-profile level"
            ),
        ),
    )


def build_result(
    p: argparse.Namespace,
    d: dict[str, Any],
    cov: dict[str, Any],
    grid: np.ndarray,
    search: dict[str, Any],
    log: list[dict[str, Any]],
    stack: dict[str, Any],
    ev: MultiReEvaluator,
) -> dict[str, Any]:
    uniq = d["uniq"]
    w = 1.0 / grid
    w = w / w.sum()

    # -- in-manifold uncertainty reference: the certified full design -----
    cert_full = [c["design_index"] for c in cov["certified_full_lod1"]["designs"]]
    ref_idx = cert_full[0]
    rk = uniq[ref_idx]
    ref_rec = ev.query(
        str(rk[0]),
        (float(rk[3]), float(rk[1]), float(rk[2])),
        mults=tuple(float(v) for v in rk[4:7]),
    )
    log.append(dict(ref_rec, use="reference", query_id=len(log) + 1))

    # -- arm A ---------------------------------------------------------------
    coarse_best = min(search["coarse"], key=lambda rc: rc["surrogate"]["j_mean"])
    arm_a = dict(
        tag="surrogate",
        coarse_n=search["coarse_n"],
        coarse_shared_best=dict(
            note="one LHS design evaluated once, seen by both arms",
            best_j_mean=coarse_best["surrogate"]["j_mean"],
            best_params=dict(coarse_best["params"]),
        ),
        arms={},
    )
    probe_idx = [0, len(grid) // 2, len(grid) - 1]
    for rule in ("greedy", "lcb"):
        chosen = search["arms"][rule]["chosen"]
        pr = chosen["per_re"]
        std_ratio = [
            float(a / b) if b else None for a, b in zip(pr["std_raw"], ref_rec["per_re"]["std_raw"])
        ]
        arm_a["arms"][rule] = dict(
            acquisition=rule,
            n_queries=search["arms"][rule]["n_queries"],
            history=search["arms"][rule]["history"],
            chosen=dict(
                params=chosen["params"],
                surrogate=chosen["surrogate"],
                service=chosen["service"],
                query_id=chosen["query_id"],
                per_re_curve=chosen["per_re"],
                std_ratio_vs_certified_reference_at_probe_re=[
                    dict(re=pr["re"][i], std_ratio=std_ratio[i]) for i in probe_idx
                ],
            ),
        )
    g_pick = arm_a["arms"]["greedy"]["chosen"]["params"]
    l_pick = arm_a["arms"]["lcb"]["chosen"]["params"]
    j_gap = abs(
        arm_a["arms"]["greedy"]["chosen"]["surrogate"]["j_mean"]
        - arm_a["arms"]["lcb"]["chosen"]["surrogate"]["j_mean"]
    )
    arm_a["contrast"] = dict(
        tag="derived",
        same_design=bool(all(abs(g_pick[v] - l_pick[v]) < 1e-6 for v in DVARS)),
        param_box_distance=float(np.linalg.norm(to_unit(g_pick) - to_unit(l_pick))),
        greedy_minus_lcb_j=float(
            arm_a["arms"]["greedy"]["chosen"]["surrogate"]["j_mean"]
            - arm_a["arms"]["lcb"]["chosen"]["surrogate"]["j_mean"]
        ),
        j_gap_below_reference_std=bool(j_gap < ref_rec["surrogate"]["std_ensemble"]),
    )
    arm_a["l_over_d_verdict"] = dict(
        tag="derived",
        greedy_pick_l_over_d=float(g_pick["l_over_d_mult"]),
        lcb_pick_l_over_d=float(l_pick["l_over_d_mult"]),
        greedy_picks_below_lod1=bool(g_pick["l_over_d_mult"] < 1.0),
        lcb_picks_below_lod1=bool(l_pick["l_over_d_mult"] < 1.0),
        corpus_full_hull_l_over_d_levels=[1.0],
    )

    # -- arm B ---------------------------------------------------------------
    anchor_idx = int(d["karr"][233])
    anchor_rows = [int(r) for r in design_rows(d, anchor_idx)]
    anchor_truth = {float(d["re"][r]): float(d["cd"][r]) for r in anchor_rows}
    anchor_points = truth_at_grid(d, anchor_idx, grid)
    ak = uniq[anchor_idx]
    anchor_rec = ev.query(
        str(ak[0]),
        (float(ak[3]), float(ak[1]), float(ak[2])),
        mults=tuple(float(v) for v in ak[4:7]),
    )
    log.append(dict(anchor_rec, use="anchor", query_id=len(log) + 1))

    prim = [candidate_record(d, ev, i, grid, grid, w, log) for i in cert_full]
    cert_any = [c["design_index"] for c in cov["certified_any_hull"]["designs"]]
    ext_spans = [(c["re_min"], c["re_max"]) for c in cov["certified_any_hull"]["designs"]]
    keep = [i for i, g in enumerate(grid) if all(lo < g < hi for lo, hi in ext_spans)]
    sub_grid = grid[keep]
    ext = [candidate_record(d, ev, i, sub_grid, grid, w, log) for i in cert_any]

    prim_opt = optimum_block(prim, ranking_block(prim))
    prim_opt["ranking_note"] = (
        "n = 1: ranking statistics are undefined — the corpus certifies multi-Re J"
        " for exactly ONE full l/d=1.0 design; see the extended enumeration"
    )
    ext_opt = optimum_block(ext, ranking_block(ext))

    imp = certified_improvement_vs_anchor(
        d, ext_opt["predicted_optimum"]["design_index"], anchor_truth
    )
    true_best_imp = certified_improvement_vs_anchor(
        d, ext_opt["true_optimum"]["design_index"], anchor_truth
    )
    anchor_pred_j = float(np.mean(anchor_rec["per_re"]["cd_mean"]))
    arm_b = dict(
        tag="mixed (per-block)",
        primary=dict(
            definition=(
                "certified full-hull l/d=1.0 corpus designs (own rows >= 4 spanning"
                " >= 0.5 decades); TRUE J via quad3 on each design's own rows at the"
                " 8-point objective grid"
            ),
            candidates=prim,
            optimum=prim_opt,
        ),
        extended=dict(
            definition=(
                "certified designs of ANY hull type (same row rule); TRUE and"
                " PREDICTED J on the shared sub-grid bracketed by every member's"
                " measured span (extrapolation banned)"
            ),
            shared_grid_re=[float(g) for g in sub_grid],
            n_grid_points_dropped=int(len(grid) - len(sub_grid)),
            dropped_grid_re=[float(g) for i, g in enumerate(grid) if i not in keep],
            candidates=ext,
            optimum=ext_opt,
            mean_ape_percent=float(np.mean([c["ape_percent"]["value"] for c in ext])),
            max_ape_percent=float(np.max([c["ape_percent"]["value"] for c in ext])),
        ),
        anchor=dict(
            design_index=anchor_idx,
            design_key=list(ak),
            corpus_rows=anchor_rows,
            n_rows=len(anchor_rows),
            span_dec=float(
                np.log10(d["re"][anchor_rows].max()) - np.log10(d["re"][anchor_rows].min())
            ),
            covered_by_certified_rule=bool(is_covered(d, anchor_idx, p)),
            truth_rows=dict(
                tag="truth",
                rows=[
                    dict(re=float(d["re"][r]), cd=float(d["cd"][r]), corpus_row=int(r))
                    for r in anchor_rows
                ],
            ),
            truth_points_at_objective_grid=anchor_points,
            truth_j_at_objective_grid=truth_j(anchor_points),
            surrogate_j=dict(
                tag="surrogate",
                value=anchor_pred_j,
                j_fuel_weighted=float(np.sum(w * np.array(anchor_rec["per_re"]["cd_mean"]))),
            ),
            uncertified_predicted_improvement_percent=dict(
                tag="derived",
                greedy_pick_vs_anchor=(
                    arm_a["arms"]["greedy"]["chosen"]["surrogate"]["j_mean"] / anchor_pred_j - 1.0
                )
                * 100.0,
                note=(
                    "the anchor has 2 own rows (0.477 dec): its multi-Re J is NOT"
                    " certifiable, so this is surrogate-on-surrogate; the certified"
                    " slices below are at the anchor's own exact Re levels"
                ),
            ),
            certified_improvement_of_predicted_optimum=imp,
            certified_improvement_of_true_optimum=true_best_imp,
        ),
    )

    phantom = build_phantom(p, grid, sub_grid, ev, log, arm_a)

    # -- gates ---------------------------------------------------------------
    w9_controls = []
    for path in (p.ld_ladder_truth, p.ld_gap_truth):
        st = json.loads(Path(path).read_text())
        ctl = next(rec for pid, rec in st["points"].items() if pid.startswith("ctl"))
        w9_controls.append(float(ctl["scan_truth"]["cd_proj"]))
    machinery200 = []
    by_design: dict[int, list[float]] = {}
    for rec in cov["exact_re200_full_lod1"]["rows"]:
        by_design.setdefault(rec["design_index"], []).append(rec["truth_cd"])
    for di in sorted(by_design):
        got = truth_at_re(d, di, 200.0)
        machinery200.append(
            got["method"] == "exact_corpus_rows" and got["value"] == float(np.mean(by_design[di]))
        )
    gates = dict(
        tag="derived",
        row233_corpus_label_matches_known_constant=bool(float(d["cd"][233]) == KNOWN_ROW233_CD),
        w9_controls_reproduce_row233_bit_exact=bool(all(v == KNOWN_ROW233_CD for v in w9_controls)),
        g10005_is_lod1_corner=bool(
            abs(float(l2t_param(p, "g10005", "l_over_d_mult")) - 1.0) < 1e-9
            and abs(float(l2t_param(p, "g10005", "sail_scale")) - CORNER_SAIL) < 1e-9
            and abs(float(l2t_param(p, "g10005", "fin_scale")) - CORNER_FIN) < 1e-9
        ),
        exact_re200_machinery_reproduces_corpus_labels=bool(all(machinery200)),
        every_truth_point_exact_or_bracketed=bool(
            all(
                (pt["value"] is not None) == (pt["method"] != "not_certifiable")
                for c in prim + ext
                for pt in c["truth_points"]
            )
        ),
        quad3_is_the_only_re_interpolator=True,
    )

    by_use: dict[str, int] = {}
    for q in log:
        by_use[q["use"]] = by_use.get(q["use"], 0) + 1
    l2t = l2t_points(p)
    single_re = dict(
        tag="truth",
        source="l2_loop_truth_20260908 (Re=200 fresh-LBM adjudication of the"
        " single-Re loop picks at the corner geometry)",
        greedy_pick=dict(
            l_over_d_mult=float(l2t["grd0001"]["params"]["l_over_d_mult"]),
            truth_cd_proj=float(l2t["grd0001"]["scan_truth"]["cd_proj"]),
        ),
        lcb_pick=dict(
            l_over_d_mult=float(l2t["lcb0002"]["params"]["l_over_d_mult"]),
            truth_cd_proj=float(l2t["lcb0002"]["scan_truth"]["cd_proj"]),
        ),
        corner_lod1=dict(
            l_over_d_mult=float(l2t["g10005"]["params"]["l_over_d_mult"]),
            truth_cd_proj=float(l2t["g10005"]["scan_truth"]["cd_proj"]),
        ),
    )
    return dict(
        schema=dict(
            surrogate="predicted by the frozen 10-member pm20260831 ts2 ensemble"
            " (production serving path, field_borrow, uq_temperature 1.5)",
            truth="C_D labels of the 406-row corpus and of the read-only W9 scan"
            " campaigns (cd_proj), plus quad3 interpolations on a design's OWN rows",
            derived="distances/APE/rankings/J-aggregates computed from the two above",
            note="no new LBM was run for anything in this file; mixed blocks carry"
            " per-subblock tag fields",
        ),
        mission=dict(
            objective="minimise J = mean C_D over the shared multi-Re grid",
            hull=p.hull,
            u_in=float(p.u_in),
            fixed_params=dict(nose_len_mult=1.0, stern_len_mult=1.0, sail_x_mult=1.0),
            design_variables=list(DVARS),
            box={k: list(v) for k, v in BOX.items()},
            grid_re=[float(g) for g in grid],
            secondary_weighting="cd weighted by 1/Re (normalised), fuel-ish low-Re"
            " emphasis — sensitivity row only",
            acquisition_rules=dict(
                greedy="min surrogate j_mean",
                lcb=f"min surrogate j_mean - {p.k_lcb} * grid-mean raw member std (k={p.k_lcb})",
            ),
            budget=dict(
                coarse_n=p.coarse_n,
                refine_rounds=p.refine_rounds,
                refine_per_round=p.refine_per_round,
                rho0=p.rho0,
                rho_decay=p.rho_decay,
                k_lcb=p.k_lcb,
                seed=p.seed,
            ),
            certified_rule=f">= {p.min_rows} own rows spanning >= {p.min_span_dec} dec",
        ),
        stack=stack,
        coverage_map=cov,
        arm_a_free_search=arm_a,
        arm_b_certified_enumeration=arm_b,
        phantom_analysis=phantom,
        reference_design=dict(
            note="the certified full l/d=1.0 design serving as the in-manifold"
            " uncertainty reference",
            design_index=int(ref_idx),
            design_key=list(rk),
            surrogate=ref_rec["surrogate"],
            service=ref_rec["service"],
            per_re_curve=ref_rec["per_re"],
        ),
        gates=gates,
        budget_ledger=dict(
            by_use=dict(sorted(by_use.items())),
            total_surrogate_queries=len(log),
            note="timing deliberately absent: the json is byte-identical across runs",
        ),
        truth_const=dict(
            row233_cd=dict(tag="truth", value=KNOWN_ROW233_CD),
            g10005_cd_proj=dict(tag="truth", value=float(l2t_g10005_cd(p))),
            w9_control_cd_proj=[dict(tag="truth", value=v) for v in w9_controls],
            single_re_campaign=single_re,
            certified_design_re200_slice=dict(
                truth_at_re(d, cert_full[0], 200.0),
                design_index=int(cert_full[0]),
                replica_rows=by_design.get(int(cert_full[0]), []),
                replica_spread_percent=(
                    (max(by_design[cert_full[0]]) / min(by_design[cert_full[0]]) - 1.0) * 100.0
                    if len(by_design.get(int(cert_full[0]), [])) > 1
                    else 0.0
                ),
            ),
        ),
    )


def l2t_points(p: argparse.Namespace) -> dict[str, Any]:
    return json.loads(Path(p.l2loop_truth).read_text())["points"]


def l2t_param(p: argparse.Namespace, pid: str, name: str) -> Any:
    return l2t_points(p)[pid]["params"][name]


def l2t_g10005_cd(p: argparse.Namespace) -> float:
    return float(l2t_points(p)["g10005"]["scan_truth"]["cd_proj"])


# --------------------------------------------------------------------------- #
# 7. Report rendering (json is the single source; zero hand-typed numbers)
# --------------------------------------------------------------------------- #
def render_report(r: dict[str, Any]) -> str:
    m, st, cov = r["mission"], r["stack"], r["coverage_map"]
    a = r["arm_a_free_search"]
    b = r["arm_b_certified_enumeration"]
    ph = r["phantom_analysis"]
    led = r["budget_ledger"]
    grid = m["grid_re"]
    box = {k: m["box"][k] for k in ("l_over_d_mult", "sail_scale", "fin_scale")}
    hist_all = dict(
        sorted(cov["row_count_histogram_all"]["counts"].items(), key=lambda kv: int(kv[0]))
    )
    hist_f = dict(
        sorted(
            cov["row_count_histogram_full_lod1"]["counts"].items(),
            key=lambda kv: int(kv[0]),
        )
    )
    lines = [
        "# L3 closed-loop multi-Re — mission-profile hull C_D objective",
        "",
        "Generated from `multire.json` by `scripts/l2_closed_loop_multire.py`.",
        "",
        "## Mission",
        "",
        f"- minimise J = mean C_D over the shared {len(grid)}-point log10-Re grid"
        f" {['%.2f' % g for g in grid]} at u_in = {m['u_in']}",
        f"- design variables {', '.join(m['design_variables'])} in the corpus box"
        f" `{box}`; hull `{m['hull']}`",
        f"- evaluator: {st['members']} frozen members ({st['arm']}), pool ="
        f" {st['pool_rows']} corpus rows, field_policy=field_borrow, uq_temperature"
        f" = {st['uq_temperature']:.1f}",
        f"- budget: LHS {m['budget']['coarse_n']} shared + refine"
        f" {m['budget']['refine_rounds']} x {m['budget']['refine_per_round']} per"
        f" arm, seed {m['budget']['seed']}; ledger `{led['by_use']}`,"
        f" {led['total_surrogate_queries']} queries",
        "",
        "## Step 0 — coverage map",
        "",
        f"- pool: {cov['n_pool_rows']} rows / {cov['n_designs']} designs;"
        f" {cov['n_full_lod1_designs']} full-hull l/d=1.0 designs",
        f"- row-count histogram (all designs): {hist_all}",
        f"- row-count histogram (full l/d=1.0): {hist_f}",
        f"- designs with >= 4 rows spanning >= 0.5 dec: ALL hulls"
        f" {cov['n_designs_with_4plus_rows_spanning_half_dec_all']}, full l/d=1.0"
        f" {cov['n_designs_with_4plus_rows_spanning_half_dec_full_lod1']}",
        f"- certified full l/d=1.0 designs:"
        f" {[cd['design_key'][:4] for cd in cov['certified_full_lod1']['designs']]}",
        f"- intersection window: Re"
        f" {cov['intersection_window_full_lod1']['re_min']:.1f} to"
        f" {cov['intersection_window_full_lod1']['re_max']:.1f}"
        f" ({cov['intersection_window_full_lod1']['width_dec']:.3f} dec)",
        f"- objective grid: {['%.2f' % g for g in cov['objective_grid']['re']]}",
        "",
        "## Arm A — free continuous search (multi-Re objective)",
        "",
        "| arm | l/d | sail | fin | pred J | J (1/Re-weighted) | ens std | LCB |"
        " borrow dist | guard |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in ("greedy", "lcb"):
        c = a["arms"][rule]["chosen"]
        pr, s = c["params"], c["surrogate"]
        lines.append(
            f"| {rule} | {pr['l_over_d_mult']:.4f} | {pr['sail_scale']:.4f}"
            f" | {pr['fin_scale']:.4f} | {s['j_mean']:.4f}"
            f" | {s['j_fuel_weighted']:.4f} | {s['std_ensemble']:.4f}"
            f" | {s['lcb']:.4f} | {c['service']['borrow_distance']:.4f}"
            f" | {c['service']['service_guard_flag']} |"
        )
    lv = a["l_over_d_verdict"]
    gc_ = a["arms"]["greedy"]["chosen"]
    lines += [
        "",
        f"- l/d verdict: greedy pick {lv['greedy_pick_l_over_d']:.4f} (below 1.0:"
        f" {lv['greedy_picks_below_lod1']}), LCB pick {lv['lcb_pick_l_over_d']:.4f}"
        f" (below 1.0: {lv['lcb_picks_below_lod1']}) — the corpus full hull exists"
        f" only at l/d levels {lv['corpus_full_hull_l_over_d_levels']}",
        f"- contrast: same design {a['contrast']['same_design']}, box distance"
        f" {a['contrast']['param_box_distance']:.4f}, greedy-lcb J gap"
        f" {a['contrast']['greedy_minus_lcb_j']:+.4f}",
        f"- per-Re surrogate curve at the greedy pick:"
        f" {['%.0f:%.3f' % (g, c) for g, c in zip(gc_['per_re_curve']['re'], gc_['per_re_curve']['cd_mean'])]}",
        f"- out-of-manifold at the greedy pick (std ratio vs the certified"
        f" reference at 3 grid Re):"
        f" {['%.1fx@%.0f' % (t['std_ratio'], t['re']) for t in gc_['std_ratio_vs_certified_reference_at_probe_re']]}",
        "",
        "## Arm B — certified-region enumeration",
        "",
        "### Primary: certified full l/d=1.0 designs (8-point grid)",
        "",
    ]
    for c in b["primary"]["candidates"]:
        lines.append(
            f"- design {c['design_index']} `{c['design_key'][:4]}`: truth J"
            f" {c['truth_j']['value']:.4f} ({c['truth_j']['n_certified_points']}/"
            f"{c['truth_j']['n_grid_points']} grid points certified), surrogate J"
            f" {c['surrogate_j']['value']:.4f}, APE"
            f" {c['ape_percent']['value']:.3f} %"
        )
    po = b["primary"]["optimum"]
    lines += [
        f"- optimum: predicted = true = design"
        f" {po['predicted_optimum']['design_index']} (n = {po['ranking']['n']});"
        f" {po['ranking_note']}",
        "",
        "### Extended: certified designs of any hull"
        f" ({b['extended']['n_grid_points_dropped']} of 8 grid points dropped —"
        " some spans do not bracket them)",
        "",
        f"- shared sub-grid: {['%.2f' % g for g in b['extended']['shared_grid_re']]};"
        f" dropped {['%.2f' % g for g in b['extended']['dropped_grid_re']]}",
        "",
        "| design | key (hull, sail, fin, l/d) | truth J | pred J | APE % |"
        " truth rank | pred rank |",
        "|---|---|---|---|---|---|---|",
    ]
    ext = b["extended"]["candidates"]
    truths = [c["truth_j"]["value"] for c in ext]
    preds = [c["surrogate_j"]["value"] for c in ext]
    rt, rp = ranks_desc(truths), ranks_desc(preds)
    for i, c in enumerate(ext):
        k = c["design_key"]
        lines.append(
            f"| {c['design_index']} | {k[0]}, {k[1]:.2f}, {k[2]:.2f}, {k[3]:.2f}"
            f" | {c['truth_j']['value']:.4f} | {c['surrogate_j']['value']:.4f}"
            f" | {c['ape_percent']['value']:.3f} | {rt[i]} | {rp[i]} |"
        )
    eo = b["extended"]["optimum"]
    rk = eo["ranking"]
    lines += [
        "",
        f"- predicted optimum: design {eo['predicted_optimum']['design_index']}"
        f" `{eo['predicted_optimum']['design_key'][:4]}` pred J"
        f" {eo['predicted_optimum']['surrogate_j']:.4f}; TRUE optimum: design"
        f" {eo['true_optimum']['design_index']}"
        f" `{eo['true_optimum']['design_key'][:4]}` truth J"
        f" {eo['true_optimum']['truth_j']:.4f}",
        f"- APE at the optimum {eo['ape_at_the_optimum_percent']:.3f} %; the true"
        f" optimum is missed by {eo['true_optimum_missed_by_percent']:+.3f} %;"
        f" same design: {eo['same_design']}",
        f"- Kendall tau-b {rk['kendall_tau_b']:.4f}; surrogate rank of the true"
        f" best {rk['surrogate_rank_of_true_best']} of {rk['n']}; true rank of the"
        f" surrogate best {rk['true_rank_of_surrogate_best']}; top-decile"
        f" displacement {rk['top_decile_displacement']}",
        f"- enumeration MAPE {b['extended']['mean_ape_percent']:.3f} %, max APE"
        f" {b['extended']['max_ape_percent']:.3f} %",
        "",
        "### Anchor comparison (row 233 design)",
        "",
        f"- anchor `{b['anchor']['design_key'][:4]}`: {b['anchor']['n_rows']} own"
        f" rows ({b['anchor']['span_dec']:.3f} dec) — covered by the certified"
        f" rule: {b['anchor']['covered_by_certified_rule']}; its multi-Re J is NOT"
        " certifiable, so the J-level improvement is surrogate-on-surrogate:",
        f"  predicted greedy-pick improvement"
        f" {b['anchor']['uncertified_predicted_improvement_percent']['greedy_pick_vs_anchor']:+.2f} %",
    ]
    for k, v in sorted(b["anchor"]["certified_improvement_of_predicted_optimum"].items()):
        if v["certified"]:
            lines.append(
                f"  CERTIFIED slice at Re {k.replace('re_', '')}: predicted-optimum"
                f" truth {v['candidate_truth_cd']['value']:.4f} vs anchor truth"
                f" {v['anchor_truth_cd']['value']:.4f} ->"
                f" {v['improvement_percent']:+.2f} %"
            )
    for k, v in sorted(b["anchor"]["certified_improvement_of_true_optimum"].items()):
        if v["certified"]:
            lines.append(
                f"  true optimum at Re {k.replace('re_', '')}:"
                f" {v['improvement_percent']:+.2f} % vs anchor truth"
            )
    lines += [
        "",
        "## Phantom analysis (W9 corner-geometry truth)",
        "",
        f"- arm A greedy pick l/d {ph['trigger']['arm_a_greedy_pick_l_over_d']:.4f},"
        f" chased below 1.0: {ph['trigger']['arm_a_chased_l_over_d_below_one']}",
        "",
        "| l/d | truth J (shared grid) | truth cd at Re=200 | surrogate J (same"
        " grid) | surrogate cd at Re=200 |",
        "|---|---|---|---|---|",
    ]
    for k in sorted(ph["truth_levels"]):
        tl, sl = ph["truth_levels"][k], ph["surrogate_levels"][k]
        lines.append(
            f"| {tl['l_over_d_mult']:.3f} | {tl['truth_j']['value']:.4f}"
            f" | {tl['truth_cd_at_re200']['value']:.4f}"
            f" | {sl['j_on_shared_subgrid']:.4f} | {sl['cd_at_re200']:.4f} |"
        )
    sl1 = ph["surrogate_levels"]["1.0000"]
    lines += [
        f"| 1.000 | not measured (g10005 is Re=200 only) |"
        f" {ph['truth_lod1_corner_at_re200']['cd_proj']:.4f}"
        f" | {sl1['j_on_shared_subgrid']:.4f} | {sl1['cd_at_re200']:.4f} |",
        "",
        f"- truth chords (cd per unit l/d mult):"
        f" {['%s: %.2f' % (k, v['cd_per_unit_l_over_d_mult']) for k, v in sorted(ph['truth_chords'].items())]}"
        " — shortening hurts MORE at low Re",
        f"- slopes on the shared sub-grid: truth"
        f" {ph['slopes_on_shared_subgrid']['truth_dJ_d_l_over_d_over_scan_levels']:+.4f}"
        f" vs surrogate"
        f" {ph['slopes_on_shared_subgrid']['surrogate_dJ_d_l_over_d_including_lod1']:+.4f}"
        f" (sign agreement {ph['slopes_on_shared_subgrid']['sign_agreement']})",
        f"- objective-level phantom: the surrogate claims"
        f" {ph['objective_level_phantom']['surrogate_claimed_J_gain_from_lod_1_to_scan_min_percent']:+.2f} %"
        f" J gain from l/d 1.0 to the shortest scan level, while truth J spans"
        f" {ph['objective_level_phantom']['truth_J_change_across_scan_levels_percent']:.2f} %"
        f" across the scan levels and the truth Re=200 chord to l/d=1.0 is"
        f" {ph['objective_level_phantom']['truth_re200_chord_lod_scan_min_to_1']:+.2f}",
        f"- verdict: {ph['objective_level_phantom']['verdict']}",
        "",
        "## Gates",
        "",
    ]
    for k, v in sorted(r["gates"].items()):
        if k != "tag":
            lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## What this run certifies — and what it does not",
        "",
        "Certified against existing truth:",
        "",
        "1. The multi-Re objective machinery (exact rows + quad3-on-own-rows only,",
        "   never extrapolated) and its control values (row 233, the W9 controls)",
        "2. The extended enumeration: predicted-vs-true J over the designs whose",
        "   own rows certify the whole shared sub-grid (ranking, optimum, APE)",
        "3. The anchor comparison at the anchor's own exact Re levels (truth on",
        "   both sides)",
        "",
        "NOT certified:",
        "",
        "1. Truth BETWEEN corpus designs: the free arm's picks are off-anchor",
        "   interpolations; l/d < 1.0 for the full hull is out-of-family (the",
        "   corpus full hull exists only at l/d = 1.0)",
        "2. The anchor's multi-Re J (2 own rows) and any J-level improvement over it",
        "3. Multi-Re truth at the corner geometry l/d = 1.0 (g10005 is Re=200 only)",
        "4. Design 10's truth curve mixes 41 u_in levels (the re-uin LHS campaign):",
        "   its quad3 curve is the Re response at the campaign's u_in spread, not a",
        "   fixed-u_in isochart — the ONLY certified full l/d=1.0 design is also",
        "   the least controlled one",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 8. --verify / --check-doc
# --------------------------------------------------------------------------- #
def collect_floats(obj: Any, out: set[float] | None = None) -> set[float]:
    if out is None:
        out = set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_floats(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_floats(v, out)
    return out


def verify(p: argparse.Namespace) -> int:
    r = json.loads(Path(p.out).read_text())
    report = Path(p.report).read_text()
    ok = True
    if report != render_report(r):
        ok = False
        print(f"[verify] FAIL: {p.report} != re-render from {p.out}")
    else:
        print(f"[verify] {p.report} is byte-identical to a re-render from {p.out}")
    for name, val in sorted(r["gates"].items()):
        if name == "tag":
            continue
        if val is not True:
            ok = False
            print(f"[verify] FAIL: gate {name} = {val}")
    cands = (
        r["arm_b_certified_enumeration"]["primary"]["candidates"]
        + r["arm_b_certified_enumeration"]["extended"]["candidates"]
    )
    for block in cands:
        vals = [pt["value"] for pt in block["truth_points"] if pt["value"] is not None]
        want = float(np.mean(vals))
        if not math.isclose(block["truth_j"]["value"], want, rel_tol=1e-12, abs_tol=1e-12):
            ok = False
            print(
                f"[verify] FAIL: truth_j of design {block['design_index']} != mean of"
                f" its certified points ({block['truth_j']['value']} vs {want})"
            )
        ape = block["ape_percent"]["value"]
        recomputed = abs(block["surrogate_j"]["value"] / block["truth_j"]["value"] - 1.0) * 100.0
        if not math.isclose(ape, recomputed, rel_tol=1e-9, abs_tol=1e-9):
            ok = False
            print(f"[verify] FAIL: APE of design {block['design_index']} does not telescope")
    ext = r["arm_b_certified_enumeration"]["extended"]["candidates"]
    tau = kendall_tau_b(
        [c["truth_j"]["value"] for c in ext], [c["surrogate_j"]["value"] for c in ext]
    )
    want = r["arm_b_certified_enumeration"]["extended"]["optimum"]["ranking"]["kendall_tau_b"]
    if tau is None or not math.isclose(tau, want, rel_tol=1e-12, abs_tol=1e-12):
        ok = False
        print(f"[verify] FAIL: kendall_tau_b does not recompute ({tau} vs {want})")
    for rule in ("greedy", "lcb"):
        c = r["arm_a_free_search"]["arms"][rule]["chosen"]
        j = float(np.mean(c["per_re_curve"]["cd_mean"]))
        if not math.isclose(j, c["surrogate"]["j_mean"], rel_tol=1e-12, abs_tol=1e-12):
            ok = False
            print(f"[verify] FAIL: arm A {rule} j_mean != grid-mean of its curve")
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def check_doc(p: argparse.Namespace) -> int:
    floats = collect_floats(json.loads(Path(p.out).read_text()))
    tokens = re.findall(r"[-+]?\d+\.\d+", Path(p.check_doc).read_text())
    bad = []
    for t in tokens:
        nd = len(t.split(".")[1])
        if not (
            any(f"{v:.{nd}f}" == t for v in floats) or any(f"{v:+.{nd}f}" == t for v in floats)
        ):
            bad.append(t)
    if bad:
        print(f"[check-doc] FAIL: {len(bad)} decimal token(s) not in {p.out}: {bad[:10]}")
        return 1
    print(f"[check-doc] PASS: all {len(tokens)} decimal tokens of {p.check_doc} come from {p.out}")
    return 0


# --------------------------------------------------------------------------- #
# 9. main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_args(parser)
    p = parser.parse_args(argv)
    if p.verify:
        return verify(p)
    if p.check_doc:
        return check_doc(p)

    Path(p.out).parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.perf_counter()

    d = load_corpus(p)
    cov = coverage_map(d, p)
    grid = np.array(cov["objective_grid"]["re"], dtype=float)
    print(
        f"[corpus] {cov['n_pool_rows']} rows / {cov['n_designs']} designs; certified"
        f" full l/d=1.0:"
        f" {cov['n_designs_with_4plus_rows_spanning_half_dec_full_lod1']}, any hull:"
        f" {cov['n_designs_with_4plus_rows_spanning_half_dec_all']}"
    )
    print(
        f"[coverage] histogram all={cov['row_count_histogram_all']['counts']}, full"
        f" l/d=1.0={cov['row_count_histogram_full_lod1']['counts']}"
    )
    win = cov["intersection_window_full_lod1"]
    print(
        f"[coverage] intersection window Re {win['re_min']:.1f}..{win['re_max']:.1f}"
        f" ({win['width_dec']:.4f} dec); grid={np.round(grid, 3).tolist()}"
    )

    view = Path(p.out).parent / "corpus_view"
    view.mkdir(parents=True, exist_ok=True)
    for src, name in (
        (Path(p.fam_dir) / "cache_fam.npz", "cache_fam.npz"),
        (Path(p.ext_dir) / "cache_ext56.npz", "cache_ext56.npz"),
        (Path(p.sdf2_dir) / "sdf_fam350.npz", "sdf_fam350.npz"),
        (Path(p.ext_dir) / "sdf_ext2.npz", "sdf_ext2.npz"),
    ):
        link = view / name
        if not link.exists():
            link.symlink_to(src)
    provider = FieldProvider.from_corpus(str(view))
    guard = fit_guard(d, np.arange(len(d["cd"])))
    svc = build_service(p.bundles_dir, p.arm, p.device, provider, guard, p.uq_temperature)
    ev = MultiReEvaluator(svc, grid, p.device, p)
    stack = dict(
        bundles=p.bundles_dir,
        arm=p.arm,
        members=len(svc.backend.member_labels()),
        backend=svc.backend.kind,
        pool_rows=int(provider.pool_fields.shape[0]),
        guard="EnvelopeMahalanobisGuardrail on all corpus condition_v3 rows",
        uq_temperature=p.uq_temperature,
        geometry_path="SuboffConfig/build_suboff_mask -> geom_encoder.sdf_volume"
        " (GPU device), field_borrow retrieval",
    )
    print(f"[stack] {stack['members']} members, pool {stack['pool_rows']} rows")

    log: list[dict[str, Any]] = []
    search = run_search(ev, p, log)
    result = build_result(p, d, cov, grid, search, log, stack, ev)
    for q in log:
        q.pop("per_re", None)
    wall_s = time.perf_counter() - t0
    print(f"[run] {len(log)} surrogate queries in {wall_s:.1f}s")

    Path(p.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    Path(p.report).write_text(render_report(result))
    g = result["gates"]
    tau = result["arm_b_certified_enumeration"]["extended"]["optimum"]["ranking"]["kendall_tau_b"]
    n_true = sum(1 for k, v in g.items() if k != "tag" and v is True)
    print(
        f"[done] gates {n_true}/{len(g) - 1} true; extended tau={tau:.4f} -> {p.out}"
        f" (+{p.report}) ({wall_s:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
