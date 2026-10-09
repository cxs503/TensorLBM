#!/usr/bin/env python3
"""L3 closed-loop claim CERTIFICATION — decompose and certify every claim of
``scripts/l2_closed_loop_demo.py`` against the accumulated campaign truth.

The 2026-09-08 closed-loop demo reported surrogate-driven C_D savings at
Re = 200 (greedy -2.01 %, LCB -1.88 % vs the corpus-best anchor row 233).
A fresh-LBM truth scan (scan_suboff_l2loop_truth_20260908, 6 points)
adjudicated those claims by hand as PHANTOMS: truth says +0.96 % / +1.93 %.
This script productizes that adjudication — no hand-typed verdicts, every
number re-derived from the read-only datasets and schema-tagged:

    leg 1  replay + per-axis decomposition — re-run the demo search by
           import (gate: picks/preds reproduce the published closed_loop.json
           bit-exactly), re-derive the scan truth cd_proj from the raw point
           artifacts (gate: the control reproduces the corpus row-233 label
           bit-exactly), attribute each arm's claim per design axis moved
           using matched-geometry truth contrasts (exact scan pairs first,
           then corpus certified-region interpolation, then the W9 l/d-axis
           scans across Re with the sanctioned quad3), and rate every
           component CERTIFIED-REFUTED / CERTIFIED-CONFIRMED / UNCERTIFIED;

    leg 2  certified-region search — re-run the optimisation restricted to
           the truth-covered designs (full hull, l/d = 1.0, corpus truth at
           exactly Re = 200): every candidate carries truth, so predicted
           optimum, TRUE optimum, ranking error and APE are all certifiable;

    leg 3  verdict + report — certify.json (surrogate/truth/derived tagged,
           byte-identical across runs) + report.md rendered FROM the json.

Truth sources (ALL read-only, no new LBM anywhere):

    /nfs/wangxi/datasets/scan_suboff_l2loop_truth_20260908  6 pts at Re=200
    /nfs/wangxi/datasets/scan_suboff_ld_anchor_20260908      l/d 0.902/0.960
    /nfs/wangxi/datasets/scan_suboff_ld_ladder_20260908      l/d 0.90/0.95
    /nfs/wangxi/datasets/scan_suboff_ld_gap_20260908         l/d 0.925/0.975
    the 406-row corpus itself (the certified region at l/d = 1.0)

Usage (5090 defaults):

    python scripts/l2_closed_loop_certify.py
    python scripts/l2_closed_loop_certify.py --truth-only     # truth gates, no GPU
    python scripts/l2_closed_loop_certify.py --verify         # render + gate recheck
    python scripts/l2_closed_loop_certify.py --check-doc docs/l2_loop_certify_20260908.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from l2_closed_loop_demo import (  # noqa: E402
    COARSE_N,
    K_LCB,
    MISSION_HULL,
    MISSION_RE,
    MISSION_U_IN,
    REFINE_PER_ROUND,
    REFINE_ROUNDS,
    RHO0,
    RHO_DECAY,
    SEED,
    Evaluator,
    run_search,
    truth_at_re,
)
from l2_serving_walkthrough import (  # noqa: E402
    build_service,
    default_paths,
    fit_guard,
    load_corpus,
)

from tensorlbm.ai.field_provider import FieldProvider  # noqa: E402
from tensorlbm.ai.inference_service import quad3_nearest3  # noqa: E402

#: Published results of the demo run (the claims under certification) —
#: /nfs/wangxi/runs/l2_closed_loop_20260908/closed_loop.json.  The replay
#: gate asserts THIS run reproduces them bit-exactly; the embedded constants
#: below are the same numbers, used only if that json is absent.
CLAIMS_JSON = "/nfs/wangxi/runs/l2_closed_loop_20260908/closed_loop.json"
PUBLISHED = {
    "greedy": {
        "params": {"l_over_d_mult": 0.919335, "sail_scale": 0.645983, "fin_scale": 2.915821},
        "cd_mean": 4.293391405802128,
        "std_ensemble": 0.08106587543395587,
        "query_id": 115,
        "claimed_percent": -2.0149201287931295,
    },
    "lcb": {
        "params": {"l_over_d_mult": 0.902405, "sail_scale": 0.600059, "fin_scale": 3.0},
        "cd_mean": 4.299501987304115,
        "std_ensemble": 0.09269724843244642,
        "query_id": 170,
        "claimed_percent": -1.8754625858067975,
    },
}

#: The hand adjudication this script must reproduce (REGRESSION GATE) —
#: truth.json of the l2_loop_truth_20260908 campaign, arms[].
#: realized_improvement_vs_control_percent.
HAND_ADJUDICATION_REALIZED_PERCENT = {"greedy": 0.9592054295797681, "lcb": 1.926031091517566}

#: The anchor of every claim: corpus row 233 = design (full, 0.4, 3.0, 1.0),
#: truth C_D 4.381678732563599.  The control point of the truth scan must
#: reproduce this value BIT-EXACTLY (the label-convention gate).
ANCHOR_CORPUS_ROW = 233
ANCHOR_TRUTH_CD = 4.381678732563599
ANCHOR_PARAMS = {"l_over_d_mult": 1.0, "sail_scale": 0.4, "fin_scale": 3.0}

#: The corner appendage geometry shared by the greedy pick and every W9
#: l/d-axis scan point.
CORNER_SAIL = 0.645983
CORNER_FIN = 2.915821

#: Scan label convention (identical to the corpus chain of row 233 and to
#: the W9 l/d-axis campaigns): tail = last 25 % of drag samples, per-point
#: projected area A = max over x of the final solid mask.
TAIL_FRAC = 0.25
U_IN = MISSION_U_IN

#: Datasets (read-only truth).
DS = {
    "l2loop": "/nfs/wangxi/datasets/scan_suboff_l2loop_truth_20260908",
    "ld_anchor": "/nfs/wangxi/datasets/scan_suboff_ld_anchor_20260908",
    "ld_ladder": "/nfs/wangxi/datasets/scan_suboff_ld_ladder_20260908",
    "ld_gap": "/nfs/wangxi/datasets/scan_suboff_ld_gap_20260908",
}

#: Design-key layout of the corpus: (hull, sail, fin, l/d, nose, stern, sail_x).
KEY_SAIL, KEY_FIN, KEY_LOD = 1, 2, 3

SCHEMA = dict(
    surrogate="predicted by the frozen 10-member pm20260831 ts2 ensemble "
    "(production serving path, field_borrow, uq_temperature 1.5)",
    truth="fresh-LBM scan labels cd_proj re-derived from the read-only dataset "
    "artifacts (drag_history tail mean + final-mask projected area) and the "
    "406-row corpus labels — nothing else",
    derived="ratios/percentages/interpolations/rankings computed from the two above",
    note="no new LBM was run for anything in this file; all datasets read-only",
)

TIER_REFUTED = "CERTIFIED-REFUTED"
TIER_CONFIRMED = "CERTIFIED-CONFIRMED"
TIER_UNCERTIFIED = "UNCERTIFIED"


def add_args(parser: argparse.ArgumentParser) -> None:
    for name, value in default_paths().items():
        parser.add_argument(f"--{name}", default=value, help=f"(default: {value})")
    parser.add_argument("--out", default="/nfs/wangxi/runs/l2_loop_certify_20260908/certify.json")
    parser.add_argument("--report", default="/nfs/wangxi/runs/l2_loop_certify_20260908/report.md")
    parser.add_argument("--claims", default=CLAIMS_JSON, help="published closed_loop.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arm", default="ts2", choices=("ts2", "ts4"))
    parser.add_argument("--uq-temperature", type=float, default=1.5)
    parser.add_argument("--re", type=float, default=MISSION_RE)
    parser.add_argument("--u-in", type=float, default=MISSION_U_IN)
    parser.add_argument("--hull", default=MISSION_HULL)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--truth-only", action="store_true", help="truth gates only, no GPU")
    parser.add_argument("--verify", action="store_true", help="re-render report + gates from json")
    parser.add_argument("--check-doc", default=None, metavar="DOC.md")


# --------------------------------------------------------------------------- #
# Truth derivation: cd_proj from the raw dataset artifacts
# --------------------------------------------------------------------------- #
def read_scan_point(ds: Path, pid: str, finite_check: bool) -> dict[str, Any]:
    """One scan point -> cd_proj, from drag_history.json + fields.h5 + status.

    Exactly the label chain of the corpus (and of the hand adjudication):
    tail = last ``TAIL_FRAC`` of the force_x samples,
    cd_proj = 2 * mean(tail Fx) / (u_in^2 * aproj),
    aproj = sum over y-z of (solid occupancy maximised over x), final step.
    """
    base = ds / "points" / pid
    st = json.loads((base / "status.json").read_text())
    if st.get("status") != "completed":
        raise RuntimeError(f"{ds.name}/{pid}: status {st.get('status')!r}, not completed")
    step = int(max(st["exported_steps"]))
    with h5py.File(base / "fields.h5", "r") as f:
        g = f[f"step_{step:06d}"]
        mask = np.asarray(g["solid_mask"])
        finite = (
            all(bool(np.isfinite(np.asarray(g[c])).all()) for c in ("ux", "uy", "uz", "rho"))
            if finite_check
            else None
        )
    dh = json.loads((base / "drag_history.json").read_text())
    fx = np.array([s["force_x"] for s in dh["samples"]], np.float64)
    tail = fx[int(fx.size * (1.0 - TAIL_FRAC)) :]
    tail_mean = float(tail.mean())
    if tail_mean != float(st["drag_mean_tail"]):
        raise RuntimeError(
            f"{ds.name}/{pid}: tail mean {tail_mean!r} != status drag_mean_tail "
            f"{st['drag_mean_tail']!r}"
        )
    aproj = int((mask > 0).max(axis=2).sum())
    pr = st["params"]
    return dict(
        point_id=pid,
        dataset=ds.name,
        params=dict(
            hull_type=str(pr["hull_type"]),
            re=float(pr["re"]),
            sail_scale=float(pr["sail_scale"]),
            fin_scale=float(pr["fin_scale"]),
            l_over_d_mult=float(pr["l_over_d_mult"]),
        ),
        truth=dict(
            tag="truth",
            tail_mean_fx=tail_mean,
            aproj=aproj,
            cd_proj=2.0 * tail_mean / (U_IN * U_IN * aproj),
            n_solid=int((mask > 0).sum()),
            step=step,
            n_drag_samples=int(fx.size),
            n_tail_samples=int(tail.size),
            fields_finite=finite,
            label_formula="cd_proj = 2*mean(tail Fx)/(u_in^2 * aproj), tail = last "
            "25% of drag samples, aproj = max-over-x of the final solid mask",
        ),
    )


def load_scan(ds_path: str, finite_check: bool) -> dict[str, dict[str, Any]]:
    """All points of one scan dataset, keyed by point id."""
    ds = Path(ds_path)
    plan = json.loads((ds / "plan.json").read_text())
    u_in = float(plan["fixed_params"]["u_in"])
    if u_in != U_IN:
        raise RuntimeError(f"{ds.name}: plan u_in {u_in} != {U_IN}")
    return {
        pdir.name: read_scan_point(ds, pdir.name, finite_check)
        for pdir in sorted((ds / "points").iterdir())
        if (pdir / "status.json").exists()
    }


def pkey(rec: dict[str, Any]) -> tuple[float, float, float]:
    """(l/d, sail, fin) of a scan point, rounded to the design-key grid."""
    pr = rec["params"]
    return tuple(round(float(pr[k]), 9) for k in ("l_over_d_mult", "sail_scale", "fin_scale"))


def pkey_of(params: dict[str, float]) -> tuple[float, float, float]:
    return tuple(round(float(params[k]), 9) for k in ("l_over_d_mult", "sail_scale", "fin_scale"))


def pct(ratio: float) -> float:
    return (ratio - 1.0) * 100.0


def tier_of(claimed_pct: float | None, truth_pct: float | None) -> str:
    """Rate one claim component by sign agreement of claim vs truth."""
    if truth_pct is None or claimed_pct is None or claimed_pct == 0.0:
        return TIER_UNCERTIFIED if truth_pct is None else TIER_CONFIRMED
    if (claimed_pct > 0.0) == (truth_pct > 0.0):
        return TIER_CONFIRMED
    return TIER_REFUTED


# --------------------------------------------------------------------------- #
# Leg 1a: the l2loop truth scan + the label-convention gate
# --------------------------------------------------------------------------- #
def truth_leg(d: dict[str, Any]) -> dict[str, Any]:
    l2loop = load_scan(DS["l2loop"], finite_check=True)
    if len(l2loop) != 6:
        raise RuntimeError(f"l2loop scan: expected 6 points, got {len(l2loop)}")
    corpus_anchor_cd = float(d["cd"][ANCHOR_CORPUS_ROW])
    by_key = {pkey(r): r for r in l2loop.values()}
    if len(by_key) != 6:
        raise RuntimeError("l2loop scan: duplicate geometries")
    control = by_key[pkey_of(ANCHOR_PARAMS)]
    return dict(
        gates=dict(
            control_reproduces_corpus_row233_bit_exact=(
                control["truth"]["cd_proj"] == corpus_anchor_cd == ANCHOR_TRUTH_CD
            ),
            all_points_fields_finite=all(r["truth"]["fields_finite"] for r in l2loop.values()),
            label_convention="cd_proj (per-point projected area, corpus chain of row 233)",
        ),
        corpus_anchor=dict(
            tag="truth",
            corpus_row=ANCHOR_CORPUS_ROW,
            params=dict(ANCHOR_PARAMS),
            truth_cd=corpus_anchor_cd,
            dataset="406-row corpus (cache_fam.npz row 233)",
        ),
        l2loop_points=dict(sorted(l2loop.items())),
        by_key=by_key,
        control_point_id=control["point_id"],
    )


def find_point(truth: dict[str, Any], lod: float, sail: float, fin: float) -> dict[str, Any]:
    key = (round(float(lod), 9), round(float(sail), 9), round(float(fin), 9))
    try:
        return truth["by_key"][key]
    except KeyError as err:
        raise KeyError(f"no l2loop scan point at geometry {key}") from err


# --------------------------------------------------------------------------- #
# Leg 1b: replay the demo search through the imported machinery
# --------------------------------------------------------------------------- #
def replay_search(
    p: argparse.Namespace, d: dict[str, Any], out_dir: Path
) -> tuple[Evaluator, dict[str, Any]]:
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    view = out_dir / "corpus_view"
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
    ev = Evaluator(svc, p.hull, p.re, p.u_in, p.device, k_lcb=K_LCB)
    sp = argparse.Namespace(
        seed=p.seed,
        coarse_n=COARSE_N,
        refine_rounds=REFINE_ROUNDS,
        refine_per_round=REFINE_PER_ROUND,
        rho0=RHO0,
        rho_decay=RHO_DECAY,
    )
    log: list[dict[str, Any]] = []
    search = run_search(ev, sp, log)
    stack = dict(
        bundles=p.bundles_dir,
        arm=p.arm,
        device=p.device,
        uq_temperature=p.uq_temperature,
        pool_rows=int(provider.pool_fields.shape[0]),
        geometry_path="the demo Evaluator: CAD -> mask -> sdf_volume -> field_borrow "
        "-> 10-member ensemble predict",
    )
    return ev, dict(search=search, n_queries=len(log), stack=stack)


def claims_source(p: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = Path(p.claims)
    if path.exists():
        return json.loads(path.read_text()), str(path)
    return dict(), "embedded published constants (claims json not found)"


def check_replay(search: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
    """Gate: the replayed picks/preds reproduce the published claims exactly."""
    out: dict[str, Any] = {}
    for rule in ("greedy", "lcb"):
        arm = search["arms"][rule]["chosen"]
        pub = PUBLISHED[rule]
        pub_claim = pub["claimed_percent"]
        pub_query_id = pub["query_id"]
        try:
            cl_arm = claims["truth_comparison"]["arms"][rule]
            pub_claim = cl_arm["claimed_improvement_vs_true_best_percent"]
            pub_query_id = cl_arm["chosen"]["query_id"]
        except (KeyError, TypeError):
            pass
        claimed_pct = (arm["surrogate"]["cd_mean"] / ANCHOR_TRUTH_CD - 1.0) * 100.0
        checks = dict(
            params=dict(
                tag="derived",
                replayed=arm["params"],
                published=dict(pub["params"]),
                equal=arm["params"] == pub["params"],
            ),
            cd_mean=dict(
                tag="derived",
                replayed=arm["surrogate"]["cd_mean"],
                published=pub["cd_mean"],
                bit_exact=arm["surrogate"]["cd_mean"] == pub["cd_mean"],
            ),
            std_ensemble=dict(
                tag="derived",
                replayed=arm["surrogate"]["std_ensemble"],
                published=pub["std_ensemble"],
                bit_exact=arm["surrogate"]["std_ensemble"] == pub["std_ensemble"],
            ),
            query_id=dict(
                tag="derived",
                replayed=arm["query_id"],
                published=pub_query_id,
                equal=arm["query_id"] == pub_query_id,
            ),
            claimed_percent=dict(
                tag="derived",
                replayed=claimed_pct,
                published=pub_claim,
                bit_exact=claimed_pct == pub_claim,
            ),
        )
        out[rule] = dict(
            replayed_pick=dict(
                tag="surrogate",
                params=arm["params"],
                surrogate=arm["surrogate"],
                service=arm["service"],
                query_id=arm["query_id"],
            ),
            reproduction_checks=checks,
            reproduced=all(v.get("equal", v.get("bit_exact", False)) for v in checks.values()),
        )
    return out


# --------------------------------------------------------------------------- #
# Leg 1c: per-axis claim decomposition
# --------------------------------------------------------------------------- #
def surrogate_query(ev: Evaluator, lod: float, sail: float, fin: float) -> dict[str, Any]:
    rec = ev.query((float(lod), float(sail), float(fin)))
    return dict(
        tag="surrogate",
        params=rec["params"],
        surrogate=rec["surrogate"],
        service=rec["service"],
    )


def contrast_row(
    axis: str,
    description: str,
    method: str,
    s_from: dict[str, Any],
    s_to: dict[str, Any],
    truth_from: float | None,
    truth_to: float | None,
    point_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One (from -> to) contrast: claimed % vs truth % and its tier."""
    claimed = pct(s_to["surrogate"]["cd_mean"] / s_from["surrogate"]["cd_mean"])
    row = dict(
        axis=axis,
        description=description,
        method=method,
        surrogate_component=dict(
            tag="surrogate",
            percent=claimed,
            from_cd=s_from["surrogate"]["cd_mean"],
            to_cd=s_to["surrogate"]["cd_mean"],
            points=point_ids or {},
        ),
    )
    if truth_from is not None and truth_to is not None:
        row["truth_component"] = dict(
            tag="truth",
            percent=pct(truth_to / truth_from),
            from_value=truth_from,
            to_value=truth_to,
            points=point_ids or {},
        )
    else:
        row["truth_component"] = None
    row["tier"] = tier_of(
        claimed, None if row["truth_component"] is None else row["truth_component"]["percent"]
    )
    return row


def certified_region_bracket(d: dict[str, Any], p: argparse.Namespace) -> dict[str, Any]:
    """Covered designs along the sail axis at fin = 3.0, l/d = 1.0, Re = 200.

    These are the only same-fin exact-truth rows of the certified region —
    the interpolation bracket for both arms' sail excursions is the TIGHTEST
    straddling pair of them (see :func:`tightest_bracket`), never a long
    chord across the whole axis.
    """
    cand = []
    for r in np.where(np.isclose(d["re"], p.re))[0]:
        k = d["uniq"][int(d["karr"][r])]
        if k[0] == p.hull and float(k[KEY_FIN]) == 3.0 and float(k[KEY_LOD]) == 1.0:
            cand.append(
                dict(
                    tag="truth",
                    sail=float(k[KEY_SAIL]),
                    design_index=int(d["karr"][r]),
                    corpus_row=int(r),
                    truth_cd=float(d["cd"][r]),
                )
            )
    cand.sort(key=lambda c: c["sail"])
    if len(cand) < 2:
        return dict(available=False, reason="no covered fin=3.0 pair at Re=200")
    return dict(
        available=True,
        method="certified_region_interpolation_between_corpus_truth (tightest covered "
        "pair straddling the excursion, linear in sail)",
        fin_scale=3.0,
        l_over_d_mult=1.0,
        covered_designs=cand,
        n_covered_fin3_designs=len({c["design_index"] for c in cand}),
    )


def tightest_bracket(covered: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    """The adjacent covered pair straddling ``target`` (tightest bracket)."""
    for lo, hi in zip(covered, covered[1:]):
        if lo["sail"] < target <= hi["sail"]:
            return dict(available=True, lo=lo, hi=hi)
    return None


def bracket_interp(bracket: dict[str, Any], target: float, field: str) -> float | None:
    """Linear (in sail) interpolation of a bracket endpoint field."""
    if not bracket.get("available"):
        return None
    lo, hi = bracket["lo"], bracket["hi"]
    if not lo["sail"] < target <= hi["sail"]:
        return None
    return lo[field] + (hi[field] - lo[field]) * ((target - lo["sail"]) / (hi["sail"] - lo["sail"]))


def decompose_arm(
    rule: str,
    ev: Evaluator,
    p: argparse.Namespace,
    truth: dict[str, Any],
    replay: dict[str, Any],
    bracket: dict[str, Any],
) -> dict[str, Any]:
    """Per-axis decomposition of one arm's claim vs the anchor."""
    pick = replay[rule]["replayed_pick"]["params"]
    anchor_pt = find_point(truth, 1.0, ANCHOR_PARAMS["sail_scale"], ANCHOR_PARAMS["fin_scale"])
    corner_pt = find_point(truth, 1.0, CORNER_SAIL, CORNER_FIN)
    pick_pt = find_point(truth, pick["l_over_d_mult"], pick["sail_scale"], pick["fin_scale"])
    anchor_truth = anchor_pt["truth"]["cd_proj"]

    s_of: dict[str, dict[str, Any]] = {}

    def served(pt: dict[str, Any]) -> dict[str, Any]:
        if pt["point_id"] not in s_of:
            s_of[pt["point_id"]] = surrogate_query(ev, *pkey(pt))
        return s_of[pt["point_id"]]

    # -- exact-chain attribution: every link a measured scan pair -----------
    if rule == "greedy":
        chain = [
            (
                "sail_scale+fin_scale",
                "sail+fin joint appendage move to the corner",
                anchor_pt,
                corner_pt,
            ),
            ("l_over_d_mult", "l/d 1.0 -> pick at the pick sail/fin", corner_pt, pick_pt),
        ]
    else:
        bridge_b = find_point(truth, 0.902, CORNER_SAIL, CORNER_FIN)
        chain = [
            (
                "sail_scale+fin_scale",
                "sail+fin joint appendage move to the corner (A)",
                anchor_pt,
                corner_pt,
            ),
            ("l_over_d_mult", "l/d 1.0 -> 0.902 at the corner appendages", corner_pt, bridge_b),
            (
                "sail_scale+fin_scale",
                "appendage move to the LCB pick at l/d 0.902 (B)",
                bridge_b,
                pick_pt,
            ),
        ]
    chain_rows = [
        contrast_row(
            axis=axis,
            description=desc,
            method="exact_scan_pair_fresh_lbm_re200",
            s_from=served(frm),
            s_to=served(to),
            truth_from=frm["truth"]["cd_proj"],
            truth_to=to["truth"]["cd_proj"],
            point_ids=dict(from_point=frm["point_id"], to_point=to["point_id"]),
        )
        for axis, desc, frm, to in chain
    ]
    telescopes = math.isclose(
        math.prod(
            lk["truth_component"]["to_value"] / lk["truth_component"]["from_value"]
            for lk in chain_rows
        ),
        pick_pt["truth"]["cd_proj"] / anchor_truth,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    # -- per-axis excursion ladder (the arm's own moves, tier-rated) --------
    exc: list[dict[str, Any]] = []
    if rule == "greedy":
        exc.append(
            contrast_row(
                axis="l_over_d_mult",
                description="l/d 1.0 -> pick l/d at the pick sail/fin (matched geometry)",
                method="exact_scan_pair_fresh_lbm_re200",
                s_from=served(corner_pt),
                s_to=served(pick_pt),
                truth_from=corner_pt["truth"]["cd_proj"],
                truth_to=pick_pt["truth"]["cd_proj"],
                point_ids=dict(from_point=corner_pt["point_id"], to_point=pick_pt["point_id"]),
            )
        )
    else:
        s_lcb_ld1 = surrogate_query(ev, 1.0, pick["sail_scale"], pick["fin_scale"])
        row = contrast_row(
            axis="l_over_d_mult",
            description="l/d 1.0 -> pick l/d at the pick sail/fin",
            method="no matched-geometry truth at sail 0.600059/fin 3.0 below l/d 1.0",
            s_from=s_lcb_ld1,
            s_to=served(pick_pt),
            truth_from=None,
            truth_to=None,
        )
        adj = next(lk for lk in chain_rows if lk["axis"] == "l_over_d_mult")
        row["adjacent_geometry_evidence"] = dict(
            tag="truth",
            note="the same l/d move measured at the corner appendages (0.645983/2.915821)",
            exact_pair_percent=adj["truth_component"]["percent"],
            from_point=adj["truth_component"]["points"]["from_point"],
            to_point=adj["truth_component"]["points"]["to_point"],
        )
        exc.append(row)

    # sail axis: certified-region interpolation (both arms), tightest bracket
    s_anchor = served(anchor_pt)
    tb = (
        tightest_bracket(bracket["covered_designs"], pick["sail_scale"])
        if bracket.get("available")
        else None
    )
    if tb is not None:
        truth_at_target = bracket_interp(tb, pick["sail_scale"], "truth_cd")
        s_lo = surrogate_query(ev, 1.0, tb["lo"]["sail"], bracket["fin_scale"])
        s_hi = surrogate_query(ev, 1.0, tb["hi"]["sail"], bracket["fin_scale"])
        pred_interp = bracket_interp(
            dict(
                available=True,
                lo=dict(sail=tb["lo"]["sail"], **{"truth_cd": s_lo["surrogate"]["cd_mean"]}),
                hi=dict(sail=tb["hi"]["sail"], **{"truth_cd": s_hi["surrogate"]["cd_mean"]}),
            ),
            pick["sail_scale"],
            "truth_cd",
        )
        claimed = pct(pred_interp / s_anchor["surrogate"]["cd_mean"])
        truth_pct = pct(truth_at_target / anchor_truth)
        bkeys = ("sail", "design_index", "corpus_row", "truth_cd")
        exc.append(
            dict(
                axis="sail_scale",
                description=(
                    f"sail {tb['lo']['sail']} -> {pick['sail_scale']} at fin 3.0, l/d 1.0 "
                    "(interpolated inside the certified region)"
                ),
                method=bracket["method"],
                surrogate_component=dict(
                    tag="surrogate",
                    percent=claimed,
                    note="the same linear interpolation applied to the served means",
                    bracket_served_lo=s_lo["surrogate"]["cd_mean"],
                    bracket_served_hi=s_hi["surrogate"]["cd_mean"],
                ),
                truth_component=dict(
                    tag="truth",
                    percent=truth_pct,
                    interpolated_truth_cd=truth_at_target,
                    bracket=[{k: tb["lo"][k] for k in bkeys}, {k: tb["hi"][k] for k in bkeys}],
                ),
                tier=tier_of(claimed, truth_pct),
            )
        )

    # fin axis: no covered contrast anywhere near sail 0.4 except the anchor
    if pick["fin_scale"] != ANCHOR_PARAMS["fin_scale"]:
        signals = dict(
            tag="derived",
            note="the honest out-of-manifold signals when truth is absent",
            ens_std_ratio_vs_anchor=(
                replay[rule]["replayed_pick"]["surrogate"]["std_ensemble"]
                / s_anchor["surrogate"]["std_ensemble"]
            ),
            borrow_distance=replay[rule]["replayed_pick"]["service"]["borrow_distance"],
            service_guard_flag=replay[rule]["replayed_pick"]["service"]["service_guard_flag"],
        )
        exc.append(
            dict(
                axis="fin_scale",
                description=(f"fin 3.0 -> {pick['fin_scale']} at sail 0.4, l/d 1.0"),
                method="no truth coverage along fin at sail 0.4 (only the anchor "
                "design itself is covered there)",
                surrogate_component=dict(tag="surrogate", percent=None),
                truth_component=None,
                tier=TIER_UNCERTIFIED,
                out_of_manifold_signals=signals,
            )
        )

    realized = pct(pick_pt["truth"]["cd_proj"] / anchor_truth)
    claimed_pct = replay[rule]["reproduction_checks"]["claimed_percent"]["replayed"]
    hand = HAND_ADJUDICATION_REALIZED_PERCENT[rule]
    pick_pred = replay[rule]["replayed_pick"]["surrogate"]["cd_mean"]
    return dict(
        rule=rule,
        pick=dict(
            tag="surrogate",
            params=pick,
            point_id=pick_pt["point_id"],
            surrogate=replay[rule]["replayed_pick"]["surrogate"],
        ),
        arm_verdict=dict(
            tag="derived",
            tier=TIER_REFUTED if realized > 0.0 else TIER_CONFIRMED,
            claimed_improvement_percent=dict(
                tag="surrogate",
                value=claimed_pct,
                note="pred C_D of the pick vs the ANCHOR TRUTH C_D (the demo mixed "
                "base: published as claimed_improvement_vs_true_best_percent)",
            ),
            realized_improvement_percent=dict(tag="truth", value=realized),
            claim_error_pp=dict(tag="derived", value=realized - claimed_pct),
            hand_adjudication_percent=dict(tag="truth", value=hand),
            reproduces_hand_adjudication=math.isclose(realized, hand, rel_tol=1e-12, abs_tol=1e-12),
            surrogate_ape_percent=dict(
                tag="derived", value=abs(pick_pred / pick_pt["truth"]["cd_proj"] - 1.0) * 100.0
            ),
        ),
        exact_chain_attribution=dict(
            tag="truth",
            note="every link is a measured scan pair; the link ratios telescope "
            "exactly to the realized total",
            links=chain_rows,
            telescoping_identity_holds=bool(telescopes),
        ),
        per_axis_excursions=exc,
        out_of_manifold_signals=dict(
            tag="derived",
            ens_std_at_pick=replay[rule]["replayed_pick"]["surrogate"]["std_ensemble"],
            ens_std_at_anchor=s_anchor["surrogate"]["std_ensemble"],
            ens_std_ratio_vs_anchor=(
                replay[rule]["replayed_pick"]["surrogate"]["std_ensemble"]
                / s_anchor["surrogate"]["std_ensemble"]
            ),
            borrow_distance=replay[rule]["replayed_pick"]["service"]["borrow_distance"],
            service_guard_flag=replay[rule]["replayed_pick"]["service"]["service_guard_flag"],
        ),
    )


# --------------------------------------------------------------------------- #
# Leg 1d: W9 l/d-axis corroboration (quad3 across Re at the corner geometry)
# --------------------------------------------------------------------------- #
def w9_corroboration(truth: dict[str, Any]) -> dict[str, Any]:
    """cd_proj(Re=200) per l/d level at the corner geometry.

    Levels of the three W9 scans get the sanctioned quad3 (nearest 3 rows in
    log10-Re, degree-2 fit) — none of them carries an exact Re=200 row.  The
    l2loop points at Re=200 are exact and double as the quad3 accuracy
    cross-check where levels coincide (l/d 0.902, 0.960).
    """
    levels: dict[float, list[tuple[float, float, str]]] = {}
    controls: dict[str, dict[str, Any]] = {}
    for name in ("ld_anchor", "ld_ladder", "ld_gap"):
        for pid, rec in sorted(load_scan(DS[name], finite_check=False).items()):
            pr = rec["params"]
            if pr["l_over_d_mult"] == 1.0 and pr["sail_scale"] == 0.4:
                controls[f"{name}:{pid}"] = dict(
                    tag="truth",
                    cd_proj=rec["truth"]["cd_proj"],
                    rel_diff_vs_corpus_anchor=rec["truth"]["cd_proj"] / ANCHOR_TRUTH_CD - 1.0,
                )
                continue
            levels.setdefault(round(pr["l_over_d_mult"], 9), []).append(
                (pr["re"], rec["truth"]["cd_proj"], f"{name}:{pid}")
            )

    base = find_point(truth, 1.0, CORNER_SAIL, CORNER_FIN)
    base_cd = base["truth"]["cd_proj"]
    rows = [
        dict(
            l_over_d_mult=1.0,
            method="exact_fresh_lbm_re200",
            point_id=base["point_id"],
            truth_cd_at_re200=dict(tag="truth", value=base_cd),
            contrast_vs_lod1_percent=None,
            source="l2loop scan (g10005)",
        )
    ]
    for lod in sorted(levels):
        pts = sorted(levels[lod])
        res = np.array([r[0] for r in pts], dtype=np.float64)
        cds = np.array([r[1] for r in pts], dtype=np.float64)
        q = quad3_nearest3(res, cds, MISSION_RE)
        if q is None:
            rows.append(
                dict(
                    l_over_d_mult=lod,
                    method="not_certifiable (no distinct 3-level bracket)",
                    truth_cd_at_re200=None,
                    contrast_vs_lod1_percent=None,
                    source="w9",
                )
            )
            continue
        rows.append(
            dict(
                l_over_d_mult=lod,
                method="quad3_across_re",
                n_re_rows=len(pts),
                re_min=float(res.min()),
                re_max=float(res.max()),
                truth_cd_at_re200=dict(tag="truth", value=float(q[0])),
                contrast_vs_lod1_percent=dict(tag="derived", value=pct(q[0] / base_cd)),
                source="w9 scans, quad3 over this level's own Re rows",
            )
        )
    for lod in (0.902, 0.919335, 0.96):
        rec = find_point(truth, lod, CORNER_SAIL, CORNER_FIN)
        rows.append(
            dict(
                l_over_d_mult=lod,
                method="exact_fresh_lbm_re200",
                point_id=rec["point_id"],
                truth_cd_at_re200=dict(tag="truth", value=rec["truth"]["cd_proj"]),
                contrast_vs_lod1_percent=dict(
                    tag="derived", value=pct(rec["truth"]["cd_proj"] / base_cd)
                ),
                source="l2loop scan",
            )
        )
    rows.sort(key=lambda r: (-r["l_over_d_mult"], r["method"]))

    quad3_checks = []
    by_lod: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_lod.setdefault(r["l_over_d_mult"], []).append(r)
    for lod, rs in sorted(by_lod.items()):
        vals = {
            r["method"]: r["truth_cd_at_re200"]["value"]
            for r in rs
            if r["truth_cd_at_re200"] is not None
        }
        if "quad3_across_re" in vals and "exact_fresh_lbm_re200" in vals:
            quad3_checks.append(
                dict(
                    l_over_d_mult=lod,
                    quad3=vals["quad3_across_re"],
                    exact=vals["exact_fresh_lbm_re200"],
                    rel_diff=vals["quad3_across_re"] / vals["exact_fresh_lbm_re200"] - 1.0,
                    rel_diff_percent=dict(
                        tag="derived",
                        value=pct(vals["quad3_across_re"] / vals["exact_fresh_lbm_re200"]),
                    ),
                )
            )
    below = [r for r in rows if r["l_over_d_mult"] < 1.0]
    covered = [r for r in below if r["contrast_vs_lod1_percent"] is not None]
    exact_cov = [r for r in covered if r["method"] == "exact_fresh_lbm_re200"]
    quad3_cov = [r for r in covered if r["method"] == "quad3_across_re"]
    ex_worse = sum(1 for r in exact_cov if r["contrast_vs_lod1_percent"]["value"] > 0.0)
    q_worse = sum(1 for r in quad3_cov if r["contrast_vs_lod1_percent"]["value"] > 0.0)
    exceptions = [
        dict(
            l_over_d_mult=r["l_over_d_mult"],
            method=r["method"],
            contrast_percent=r["contrast_vs_lod1_percent"]["value"],
        )
        for r in covered
        if r["contrast_vs_lod1_percent"]["value"] <= 0.0
    ]
    max_quad3_rel_err = max(abs(qc["rel_diff"]) for qc in quad3_checks) if quad3_checks else None
    return dict(
        geometry=dict(sail_scale=CORNER_SAIL, fin_scale=CORNER_FIN),
        note="l/d axis of the corner geometry: cd_proj at Re=200 per level; a "
        "positive contrast means moving below l/d 1.0 INCREASES truth C_D",
        rows=rows,
        quad3_vs_exact_cross_checks=quad3_checks,
        max_abs_quad3_rel_diff_vs_exact=dict(tag="derived", value=max_quad3_rel_err)
        if max_quad3_rel_err is not None
        else None,
        control_replicas=controls,
        n_exact_rows_below_1=len(exact_cov),
        n_exact_rows_below_1_truth_worse=ex_worse,
        n_quad3_rows_below_1=len(quad3_cov),
        n_quad3_rows_below_1_truth_worse=q_worse,
        exceptions_below_1_not_worse=exceptions,
        verdict=(
            f"the l/d axis below 1.0 is REFUTED at this geometry: every exact "
            f"fresh-LBM row below 1.0 ({ex_worse}/{len(exact_cov)}) and "
            f"{q_worse}/{len(quad3_cov)} quad3 levels sit above the l/d 1.0 truth "
            "at Re=200"
            + (
                "; sole exception: the quad3-only estimate at l/d "
                f"{exceptions[0]['l_over_d_mult']} "
                f"({exceptions[0]['contrast_percent']:+.4f} %, no exact Re=200 row) "
                "— while both arms' actual excursions (l/d 0.919 / 0.902) sit deep "
                "inside the refuted region"
                if exceptions
                else ""
            )
        ),
    )


# --------------------------------------------------------------------------- #
# Leg 2: certified-region search (every candidate carries corpus truth)
# --------------------------------------------------------------------------- #
def leg2_search(ev: Evaluator, d: dict[str, Any], p: argparse.Namespace) -> dict[str, Any]:
    """Enumerate the truth-covered designs through the serving path.

    The certified region at the mission Re is DISCRETE — the full-hull
    l/d = 1.0 corpus designs whose rows include exactly Re = 200 (the others
    neither have that row nor bracket it, so quad3 cannot certify them).
    Enumerating that set IS the constrained optimisation: an exact optimiser
    over a discrete feasible set, every query the production serving path.
    """
    design_rows: dict[int, list[int]] = {}
    for r in np.where(np.isclose(d["re"], p.re))[0]:
        k = d["uniq"][int(d["karr"][r])]
        if k[0] == p.hull and float(k[KEY_LOD]) == 1.0:
            design_rows.setdefault(int(d["karr"][r]), []).append(int(r))
    n_full_ld1 = sum(1 for k in d["uniq"] if k[0] == p.hull and float(k[KEY_LOD]) == 1.0)
    rows = []
    for di in sorted(design_rows):
        k = d["uniq"][di]
        t = truth_at_re(d, di, p.re)
        if t["method"] != "exact_corpus_rows":
            continue
        rec = ev.query((float(k[KEY_LOD]), float(k[KEY_SAIL]), float(k[KEY_FIN])))
        rows.append(
            dict(
                design_index=di,
                design_key=list(k),
                truth_corpus_rows=[int(r) for r in sorted(design_rows[di])],
                truth_cd_at_re200=dict(tag="truth", value=float(t["value"])),
                surrogate=rec["surrogate"],
                service=rec["service"],
            )
        )
    truth_order = sorted(rows, key=lambda r: r["truth_cd_at_re200"]["value"])
    pred_order = sorted(rows, key=lambda r: r["surrogate"]["cd_mean"])
    lcb_order = sorted(rows, key=lambda r: r["surrogate"]["lcb"])
    t_rank = {r["design_index"]: i + 1 for i, r in enumerate(truth_order)}
    s_rank = {r["design_index"]: i + 1 for i, r in enumerate(pred_order)}
    n = len(rows)
    n_top = max(1, math.ceil(0.1 * n))
    truth_top = {r["design_index"] for r in truth_order[:n_top]}
    pred_top = {r["design_index"] for r in pred_order[:n_top]}
    anchor = truth_order[0]

    def pick_block(order: list[dict[str, Any]], rule: str) -> dict[str, Any]:
        pk = order[0]
        return dict(
            rule=rule,
            design_index=pk["design_index"],
            design_key=pk["design_key"],
            surrogate=pk["surrogate"],
            truth_cd_at_re200=pk["truth_cd_at_re200"],
            surrogate_rank=s_rank[pk["design_index"]],
            ape_percent=dict(
                tag="derived",
                value=abs(pk["surrogate"]["cd_mean"] / pk["truth_cd_at_re200"]["value"] - 1.0)
                * 100.0,
            ),
            certified_improvement_vs_region_true_best_percent=dict(
                tag="truth",
                value=pct(pk["truth_cd_at_re200"]["value"] / anchor["truth_cd_at_re200"]["value"]),
            ),
            pick_is_true_best=pk["design_index"] == anchor["design_index"],
        )

    return dict(
        region_definition=dict(
            hull=p.hull,
            l_over_d_mult=1.0,
            re=float(p.re),
            truth_coverage="corpus rows at exactly Re = 200 (the corpus cd label chain of row 233)",
            n_full_ld1_designs_in_corpus=n_full_ld1,
            n_candidates=n,
            optimiser="exact enumeration of the discrete truth-covered set "
            "(greedy = min served mean, lcb = min served mean - 1.0 * ensemble std)",
            note="the other full l/d=1.0 designs have no exact Re=200 row and "
            "their own curves do not bracket Re=200, so quad3 cannot certify "
            "them; certifiable continuous search inside the region would need "
            "truth BETWEEN designs, which Leg 1 refuted the surrogate at",
        ),
        candidates=[
            dict(
                r,
                truth_rank=t_rank[r["design_index"]],
                surrogate_rank=s_rank[r["design_index"]],
                ape_percent=dict(
                    tag="derived",
                    value=abs(r["surrogate"]["cd_mean"] / r["truth_cd_at_re200"]["value"] - 1.0)
                    * 100.0,
                ),
            )
            for r in sorted(rows, key=lambda r: r["truth_cd_at_re200"]["value"])
        ],
        true_optimum=dict(
            design_index=anchor["design_index"],
            design_key=anchor["design_key"],
            truth_cd=anchor["truth_cd_at_re200"],
            note="the row-233 basin: the anchor design is the TRUE optimum of "
            "the certified region at Re=200",
        ),
        picks=dict(greedy=pick_block(pred_order, "greedy"), lcb=pick_block(lcb_order, "lcb")),
        ranking=dict(
            tag="derived",
            top_decile_size=n_top,
            truth_top_decile=sorted(truth_top),
            surrogate_top_decile=sorted(pred_top),
            displaced_from_top_decile=sorted(truth_top - pred_top),
            n_displaced=len(truth_top - pred_top),
            kendall_tau_truth_vs_surrogate=kendall_tau(
                np.array([t_rank[r["design_index"]] for r in rows]),
                np.array([s_rank[r["design_index"]] for r in rows]),
            ),
            surrogate_rank_of_true_best=s_rank[anchor["design_index"]],
            ape_of_surrogate_at_true_best=dict(
                tag="derived",
                value=abs(
                    anchor["surrogate"]["cd_mean"] / anchor["truth_cd_at_re200"]["value"] - 1.0
                )
                * 100.0,
            ),
        ),
    )


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (0.5 * n * (n - 1))


# --------------------------------------------------------------------------- #
# Leg 3: verdict + assembly + report rendering (json is the single source)
# --------------------------------------------------------------------------- #
def build_verdict(arms: dict[str, Any], leg2: dict[str, Any], w9: dict[str, Any]) -> dict:
    phantoms = {
        rule: dict(
            tag="derived",
            claimed_percent=arms[rule]["arm_verdict"]["claimed_improvement_percent"]["value"],
            realized_percent=arms[rule]["arm_verdict"]["realized_improvement_percent"]["value"],
            tier=arms[rule]["arm_verdict"]["tier"],
        )
        for rule in arms
    }
    pk = leg2["picks"]["greedy"]
    return dict(
        phantom_verdicts=dict(
            tag="derived",
            summary=(
                "both closed-loop claims are PHANTOMS: the surrogate claimed "
                "savings where fresh LBM measures regressions"
            ),
            arms=phantoms,
        ),
        certified_region_answer=dict(
            tag="derived",
            predicted_optimum_design_index=pk["design_index"],
            predicted_optimum_truth_cd=pk["truth_cd_at_re200"]["value"],
            true_optimum_design_index=leg2["true_optimum"]["design_index"],
            true_optimum_truth_cd=leg2["true_optimum"]["truth_cd"]["value"],
            certified_improvement_percent=pk["certified_improvement_vs_region_true_best_percent"][
                "value"
            ],
            ranking_error_n_displaced=leg2["ranking"]["n_displaced"],
            ape_percent=pk["ape_percent"]["value"],
        ),
        what_certification_cannot_cover=[
            "anything outside the accumulated truth: the demo search box spans "
            "l/d 0.75-1.30 while same-hull truth exists only at l/d 1.0 (corpus) "
            "plus the scanned bridge points",
            "the l/d axis below 1.0 is REFUTED, not merely uncertified, at the "
            "covered geometries: " + w9["verdict"],
            "truth BETWEEN corpus designs inside the certified region is not "
            "certified either — Leg 2 therefore searches the discrete covered "
            "design set, and its 0 % certified improvement is the honest ceiling "
            "of what this truth can certify at Re=200",
            "the fin axis at sail 0.4 has no covered contrast; the greedy arm "
            "fin excursion stays UNCERTIFIED (out-of-manifold signals attached)",
            "single-Re (200), single-objective (C_D), in-family box, hull full; "
            "no new LBM was run for this certification",
        ],
    )


def build_result(
    p: argparse.Namespace,
    truth: dict[str, Any],
    replay_checks: dict[str, Any],
    arms: dict[str, Any],
    w9: dict[str, Any],
    leg2: dict[str, Any],
    stack: dict[str, Any],
    claims_src: str,
    bracket: dict[str, Any],
) -> dict[str, Any]:
    verdict = build_verdict(arms, leg2, w9)
    gates = dict(
        tag="derived",
        replay_reproduces_published_claims=all(r["reproduced"] for r in replay_checks.values()),
        control_reproduces_corpus_row233_bit_exact=truth["gates"][
            "control_reproduces_corpus_row233_bit_exact"
        ],
        all_scan_points_fields_finite=truth["gates"]["all_points_fields_finite"],
        hand_adjudication_reproduced=all(
            arms[r]["arm_verdict"]["reproduces_hand_adjudication"] for r in arms
        ),
        exact_chains_telescope=all(
            arms[r]["exact_chain_attribution"]["telescoping_identity_holds"] for r in arms
        ),
        w9_control_replicas_bit_exact=all(
            v["rel_diff_vs_corpus_anchor"] == 0.0 for v in w9["control_replicas"].values()
        ),
    )
    return dict(
        schema=SCHEMA,
        mission=dict(
            objective="certify the claims of scripts/l2_closed_loop_demo.py",
            hull=p.hull,
            re=float(p.re),
            u_in=float(p.u_in),
            anchor=dict(
                corpus_row=ANCHOR_CORPUS_ROW,
                params=dict(ANCHOR_PARAMS),
                truth_cd=dict(tag="truth", value=ANCHOR_TRUTH_CD),
                role="the corpus-best design at Re=200 and the baseline of every claimed saving",
            ),
            claims_source=claims_src,
            truth_datasets=dict(DS),
            search_replay_budget=dict(
                coarse_n=COARSE_N,
                refine_rounds=REFINE_ROUNDS,
                refine_per_round=REFINE_PER_ROUND,
                k_lcb=K_LCB,
                rho0=RHO0,
                rho_decay=RHO_DECAY,
                seed=p.seed,
            ),
        ),
        stack=stack,
        gates=gates,
        truth_derivation=dict(
            label_convention=truth["gates"]["label_convention"],
            control_point_id=truth["control_point_id"],
            l2loop_points=truth["l2loop_points"],
        ),
        leg1_replay=replay_checks,
        leg1_sail_bracket=bracket,
        leg1_decomposition=arms,
        leg1_w9_ld_axis=w9,
        leg2_certified_region_search=leg2,
        leg3_verdict=verdict,
    )


def cell(v: float | None, fmtstr: str) -> str:
    return "n/a" if v is None else format(v, fmtstr)


def render_report(cert: dict[str, Any]) -> str:
    """report.md FROM the certify dict — zero hand-typed numbers."""
    m, g = cert["mission"], cert["gates"]
    lines = [
        "# L3 closed-loop claim certification — the phantom adjudication, productized",
        "",
        "Generated from `certify.json` by `scripts/l2_closed_loop_certify.py`.",
        "",
        "## Mission and gates",
        "",
        f"- claims under certification: the {m['hull']}-hull Re = {m['re']:.1f} picks of"
        f" scripts/l2_closed_loop_demo.py vs the anchor (corpus row"
        f" {m['anchor']['corpus_row']}, truth C_D {m['anchor']['truth_cd']['value']:.6f})",
        f"- claims source `{m['claims_source']}`; truth datasets:"
        f" {', '.join(sorted(m['truth_datasets']))} (read-only, no new LBM)",
        f"- gates: replay bit-exact = {g['replay_reproduces_published_claims']}, control"
        f" bit-exact = {g['control_reproduces_corpus_row233_bit_exact']}, hand"
        f" adjudication reproduced = {g['hand_adjudication_reproduced']}, chains"
        f" telescope = {g['exact_chains_telescope']}, W9 controls bit-exact ="
        f" {g['w9_control_replicas_bit_exact']}, scan fields finite ="
        f" {g['all_scan_points_fields_finite']}",
        "",
        "## Leg 1 — phantom verdicts",
        "",
        "| arm | claimed % (surrogate) | realized % (fresh LBM) | error pp | tier |",
        "|---|---|---|---|---|",
    ]
    for rule in ("greedy", "lcb"):
        a = cert["leg1_decomposition"][rule]["arm_verdict"]
        lines.append(
            f"| {rule} | {a['claimed_improvement_percent']['value']:+.4f}"
            f" | {a['realized_improvement_percent']['value']:+.4f}"
            f" | {a['claim_error_pp']['value']:+.4f} | **{a['tier']}** |"
        )
    lines += [
        "",
        "Both claims are PHANTOMS. Reproduction of the hand adjudication"
        " (realized vs the hand value):",
    ]
    for rule in ("greedy", "lcb"):
        a = cert["leg1_decomposition"][rule]["arm_verdict"]
        lines.append(
            f"- {rule}: {a['realized_improvement_percent']['value']:+.9f} vs hand"
            f" {a['hand_adjudication_percent']['value']:+.9f}"
            f" (reproduced = {a['reproduces_hand_adjudication']}, surrogate APE at the"
            f" pick {a['surrogate_ape_percent']['value']:.4f} %)"
        )
    lines += [
        "",
        "### Per-axis attribution — exact scan chains (each link a measured pair)",
        "",
    ]
    for rule in ("greedy", "lcb"):
        arm = cert["leg1_decomposition"][rule]
        pp = arm["pick"]["params"]
        lines += [
            f"**{rule} pick** `{arm['pick']['point_id']}`"
            f" (l/d {pp['l_over_d_mult']:.6f}, sail {pp['sail_scale']:.6f},"
            f" fin {pp['fin_scale']:.6f}):",
            "",
            "| chain link | surrogate % | truth % | tier |",
            "|---|---|---|---|",
        ]
        for lk in arm["exact_chain_attribution"]["links"]:
            lines.append(
                f"| {lk['description']} | {lk['surrogate_component']['percent']:+.4f}"
                f" | {lk['truth_component']['percent']:+.4f} | {lk['tier']} |"
            )
        lines.append("")
    lines += [
        "### Per-axis excursion ladder (the arm's own moves, tier-rated)",
        "",
        "| arm | axis | surrogate % | truth % | method | tier |",
        "|---|---|---|---|---|---|",
    ]
    for rule in ("greedy", "lcb"):
        for exc in cert["leg1_decomposition"][rule]["per_axis_excursions"]:
            sp = exc["surrogate_component"]["percent"]
            tp = None if exc["truth_component"] is None else exc["truth_component"]["percent"]
            lines.append(
                f"| {rule} | {exc['axis']} | {cell(sp, '+.4f')} | {cell(tp, '+.4f')}"
                f" | {exc['method']} | {exc['tier']} |"
            )
    lines += ["", "Adjacent-geometry evidence for the UNCERTIFIED l/d rows:"]
    for rule in ("greedy", "lcb"):
        for exc in cert["leg1_decomposition"][rule]["per_axis_excursions"]:
            adj = exc.get("adjacent_geometry_evidence")
            if adj:
                lines.append(
                    f"- {rule}: the same l/d move at the corner appendages measures"
                    f" {adj['exact_pair_percent']:+.4f} % truth"
                    f" (`{adj['from_point']}` -> `{adj['to_point']}`)"
                )
    lines += [
        "",
        "### Out-of-manifold signals at the picks (the honest UNCERTIFIED signals)",
        "",
        "| arm | ens std | ratio vs anchor | borrow dist | guard |",
        "|---|---|---|---|---|",
    ]
    for rule in ("greedy", "lcb"):
        o = cert["leg1_decomposition"][rule]["out_of_manifold_signals"]
        lines.append(
            f"| {rule} | {o['ens_std_at_pick']:.4f} | {o['ens_std_ratio_vs_anchor']:.2f}"
            f" | {o['borrow_distance']:.3f} | {o['service_guard_flag']} |"
        )
    w9 = cert["leg1_w9_ld_axis"]
    lines += [
        "",
        "### W9 l/d-axis corroboration (corner geometry, quad3 across Re)",
        "",
        f"{w9['note']}.",
        "",
        "| l/d | cd_proj at Re=200 | contrast vs l/d 1.0 % | method |",
        "|---|---|---|---|",
    ]
    for r in w9["rows"]:
        cdv = r["truth_cd_at_re200"]
        ct = r["contrast_vs_lod1_percent"]
        lines.append(
            f"| {r['l_over_d_mult']:.6f} |"
            f" {cell(None if cdv is None else cdv['value'], '.6f')} |"
            f" {cell(None if ct is None else ct['value'], '+.4f')} | {r['method']} |"
        )
    for qc in w9["quad3_vs_exact_cross_checks"]:
        lines.append(
            f"- quad3 vs exact cross-check at l/d {qc['l_over_d_mult']:.3f}:"
            f" {qc['quad3']:.6f} vs {qc['exact']:.6f} (rel diff"
            f" {qc['rel_diff_percent']['value']:+.4f} %)"
        )
    lines += ["", f"- {w9['verdict']}", ""]
    leg2 = cert["leg2_certified_region_search"]
    lines += [
        "## Leg 2 — certified-region search (every candidate carries truth)",
        "",
        f"- region: {leg2['region_definition']['n_candidates']} covered designs (full"
        f" hull, l/d 1.0, exact Re = {leg2['region_definition']['re']:.1f} corpus rows) of"
        f" {leg2['region_definition']['n_full_ld1_designs_in_corpus']} full l/d=1.0"
        " designs in the corpus — the discrete truth-covered set at the mission Re;",
        f" optimiser: {leg2['region_definition']['optimiser']}",
        "",
        "| arm | pick design | pick (sail, fin) | pred C_D | truth C_D | APE % |",
        "|---|---|---|---|---|---|",
    ]
    for rule in ("greedy", "lcb"):
        pk = leg2["picks"][rule]
        k = pk["design_key"]
        lines.append(
            f"| {rule} | {pk['design_index']} | ({k[KEY_SAIL]:.6f}, {k[KEY_FIN]:.6f})"
            f" | {pk['surrogate']['cd_mean']:.6f}"
            f" | {pk['truth_cd_at_re200']['value']:.6f}"
            f" | {pk['ape_percent']['value']:.4f} |"
        )
    rk = leg2["ranking"]
    lines += [
        "",
        f"- TRUE optimum of the region: design {leg2['true_optimum']['design_index']}"
        f" (`{leg2['true_optimum']['design_key']}`) truth"
        f" {leg2['true_optimum']['truth_cd']['value']:.6f} — the row-233 basin;",
        f"- certified improvement of the greedy pick vs the region TRUE best:"
        f" {leg2['picks']['greedy']['certified_improvement_vs_region_true_best_percent']['value']:+.4f} %"
        " (0 % means the pick IS the in-region optimum — no in-region saving exists)",
        f"- ranking error: {rk['n_displaced']} of the truth top decile"
        f" ({rk['top_decile_size']} designs) displaced by the surrogate ranking;"
        f" Kendall tau = {rk['kendall_tau_truth_vs_surrogate']:.4f}; surrogate rank of"
        f" the true best = {rk['surrogate_rank_of_true_best']}, APE there"
        f" {rk['ape_of_surrogate_at_true_best']['value']:.4f} %",
        "",
        "## Leg 3 — verdict",
        "",
        f"- {cert['leg3_verdict']['phantom_verdicts']['summary']}.",
    ]
    pv = cert["leg3_verdict"]["phantom_verdicts"]["arms"]
    for rule in ("greedy", "lcb"):
        lines.append(
            f"- {rule}: claimed {pv[rule]['claimed_percent']:+.4f} %, truth"
            f" {pv[rule]['realized_percent']:+.4f} % ({pv[rule]['tier']})"
        )
    cr = cert["leg3_verdict"]["certified_region_answer"]
    lines += [
        f"- certified-region answer: predicted optimum design"
        f" {cr['predicted_optimum_design_index']} (truth"
        f" {cr['predicted_optimum_truth_cd']:.6f}), TRUE optimum design"
        f" {cr['true_optimum_design_index']} (truth"
        f" {cr['true_optimum_truth_cd']:.6f}), certified improvement"
        f" {cr['certified_improvement_percent']:+.4f} %, ranking error"
        f" {cr['ranking_error_n_displaced']}, APE {cr['ape_percent']:.4f} %",
        "",
        "## What this certification cannot cover",
        "",
    ]
    lines += [
        f"{i + 1}. {s}"
        for i, s in enumerate(cert["leg3_verdict"]["what_certification_cannot_cover"])
    ]
    lines.append("")
    return "\n".join(lines)


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
    cert = json.loads(Path(p.out).read_text())
    report = Path(p.report).read_text()
    ok = True
    if report != render_report(cert):
        ok = False
        print(f"[verify] FAIL: {p.report} != re-render from {p.out}")
    else:
        print(f"[verify] {p.report} is byte-identical to a re-render from {p.out}")
    for name, val in sorted(cert["gates"].items()):
        if name == "tag":
            continue
        if val is not True:
            ok = False
            print(f"[verify] FAIL: gate {name} = {val}")
    for rule in ("greedy", "lcb"):
        arm = cert["leg1_decomposition"][rule]
        prod = math.prod(
            lk["truth_component"]["to_value"] / lk["truth_component"]["from_value"]
            for lk in arm["exact_chain_attribution"]["links"]
        )
        want = 1.0 + arm["arm_verdict"]["realized_improvement_percent"]["value"] / 100.0
        if not math.isclose(prod, want, rel_tol=1e-12, abs_tol=1e-12):
            ok = False
            print(f"[verify] FAIL: {rule} chain does not telescope ({prod} vs {want})")
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def check_doc(p: argparse.Namespace) -> int:
    cert = json.loads(Path(p.out).read_text())
    floats = collect_floats(cert)
    tokens = re.findall(r"[-+]?\d+\.\d+", Path(p.check_doc).read_text())
    bad = []
    for t in tokens:
        nd = len(t.split(".")[1])
        if not (
            any(f"{v:.{nd}f}" == t for v in floats) or any(f"{v:+.{nd}f}" == t for v in floats)
        ):
            bad.append(t)
    if bad:
        print(f"[check-doc] FAIL: {len(bad)} decimal token(s) not in certify.json: {bad[:10]}")
        return 1
    print(f"[check-doc] PASS: all {len(tokens)} decimal tokens of {p.check_doc} come from {p.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_args(parser)
    p = parser.parse_args(argv)
    if p.verify:
        return verify(p)
    if p.check_doc:
        return check_doc(p)

    d = load_corpus(p)
    truth = truth_leg(d)
    print(
        f"[truth] l2loop 6 points re-derived; control {truth['control_point_id']}"
        f" bit-exact vs corpus row {ANCHOR_CORPUS_ROW}:"
        f" {truth['gates']['control_reproduces_corpus_row233_bit_exact']}"
    )
    if not truth["gates"]["control_reproduces_corpus_row233_bit_exact"]:
        print("[BLOCKER] control point does not reproduce the corpus label bit-exactly")
        return 2
    w9 = w9_corroboration(truth)
    print(
        f"[w9] l/d rows below 1.0 with truth worse:"
        f" {w9['n_exact_rows_below_1_truth_worse']}/{w9['n_exact_rows_below_1']} exact,"
        f" {w9['n_quad3_rows_below_1_truth_worse']}/{w9['n_quad3_rows_below_1']} quad3"
        f" (+{w9['n_quad3_rows_below_1_truth_worse']}/{w9['n_quad3_rows_below_1']} quad3);"
        " control replicas:"
        f" {sorted(w9['control_replicas'])}"
    )
    if p.truth_only:
        print("[truth-only] stopping before the GPU legs")
        return 0

    out_dir = Path(p.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ev, rep = replay_search(p, d, out_dir)
    search = rep["search"]
    print(f"[replay] {rep['n_queries']} surrogate queries")
    claims, claims_src = claims_source(p)
    replay_checks = check_replay(search, claims)
    for rule in ("greedy", "lcb"):
        rc = replay_checks[rule]
        print(
            f"[replay:{rule}] reproduced={rc['reproduced']} pick"
            f" {rc['replayed_pick']['params']} cd="
            f"{rc['replayed_pick']['surrogate']['cd_mean']:.6f}"
        )
        if not rc["reproduced"]:
            print(f"[BLOCKER] {rule} arm does not reproduce the published pick bit-exactly")
            return 2

    bracket = certified_region_bracket(d, p)
    arms = {
        rule: decompose_arm(rule, ev, p, truth, replay_checks, bracket)
        for rule in ("greedy", "lcb")
    }
    for rule in ("greedy", "lcb"):
        av = arms[rule]["arm_verdict"]
        print(
            f"[adjudicate:{rule}] claimed {av['claimed_improvement_percent']['value']:+.4f}%"
            f" -> realized {av['realized_improvement_percent']['value']:+.4f}%"
            f" ({av['tier']}), hand-reproduction {av['reproduces_hand_adjudication']}"
        )
        if not av["reproduces_hand_adjudication"]:
            print(f"[BLOCKER] {rule} realized improvement does not reproduce the hand value")
            return 2

    leg2 = leg2_search(ev, d, p)
    print(
        f"[leg2] {leg2['region_definition']['n_candidates']} certified candidates;"
        f" greedy pick design {leg2['picks']['greedy']['design_index']}"
        f" (true best {leg2['true_optimum']['design_index']}), displaced-in-top-decile"
        f" {leg2['ranking']['n_displaced']}"
    )

    cert = build_result(p, truth, replay_checks, arms, w9, leg2, rep["stack"], claims_src, bracket)
    Path(p.out).write_text(json.dumps(cert, indent=1, sort_keys=True))
    Path(p.report).write_text(render_report(cert))
    print(f"[done] gates {json.dumps(cert['gates'], sort_keys=True)} -> {p.out} (+{p.report})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
