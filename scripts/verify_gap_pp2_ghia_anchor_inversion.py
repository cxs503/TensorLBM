#!/usr/bin/env python
"""GAP-PP-2-lid-wall-diff · external-anchor (Ghia 1982) ``tau`` inversion.

Why this script exists
----------------------
``docs/GAP_PP_2_LID_WALL_DIFF.md`` §8.3 closes acceptance criterion ①
(``tau`` recovery <2 %) with a *same-chain* inversion: the observations come
from the very differentiable chain that is then inverted.  That is a
correctness check of the gradients, **not** an inversion of measured data.
The non-circular reading is to invert against data the model never produced --
the Ghia/Ghia/Shin (1982) Re=100 centreline tables -- and to measure what the
model prefers instead.

What it computes
----------------
1. ``steady`` roll-outs of the graph-preserving lid chain
   (:mod:`tensorlbm.diff_lid_cavity`) to a residual tolerance, on a ``Re``
   ladder (``tau = 1/2 + 3 u_lid H / Re``, ``H = n-1``);
2. the external loss ``J(Re) = rmse_u^2 + rmse_v^2`` against the Ghia tables,
   split into its ``u`` (vertical centreline) and ``v`` (horizontal) parts, and
   evaluated under three *node -> physical coordinate* mappings, because
   ``docs/GHIA_METRIC_CONVENTION_AUDIT.md`` showed the mapping alone moves the
   RMSE by O(h):

   ``node``    ``y_j = j/(n-1)``                  (legacy default)
   ``half``    ``y_j = (j+0.5)/n``                (cells centred, audited w=+0.5)
   ``halfbb``  ``y_j = (j-0.5)/(n-1.5)``          (half-way bounce-back walls
                                                  at the solid node row, lid on
                                                  the top node row)

   All three use only the *interior* Ghia rows (``0 < y < 1``, ``0 < x < 1``),
   so the comparison is not contaminated by the wall rows that ``np.interp``
   silently clamps.  The legacy ``compare_ghia`` numbers (all rows, w=0) are
   reported alongside for continuity with the previous documents;
3. a quadratic surrogate of ``J(Re)`` -> the *anchor* Reynolds number the
   external data prefers, and the induced ``tau`` bias;
4. the same ladder at ``u_lid = 0.05`` (half the Mach number) and at ``n = 129``
   (twice the resolution), to attribute the bias to finite-Mach
   compressibility vs. discretisation.

Every number is written to JSON by the code path that computed it.

Usage (sharded on the cluster)::

    PYTHONPATH=src python scripts/verify_gap_pp2_ghia_anchor_inversion.py \\
        shard --index 1/13 --outdir artifacts/gap_pp2_ghia_anchor
    PYTHONPATH=src python scripts/verify_gap_pp2_ghia_anchor_inversion.py \\
        collect --outdir artifacts/gap_pp2_ghia_anchor
    PYTHONPATH=src python scripts/verify_gap_pp2_ghia_anchor_inversion.py \\
        figures --outdir artifacts/gap_pp2_ghia_anchor
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from tensorlbm.d2q9 import macroscopic
from tensorlbm.diff_lid_cavity import (
    CavityGeometry,
    initial_field,
    step,
    tau_from_reynolds,
)
from tensorlbm.lid_driven_cavity import GHIA_RE100, compare_ghia

DTYPE = torch.float64
RE_REF = 100.0  # the Ghia table we invert against

# (n, u_lid) groups; each gets its own Re ladder.  u_lid=0.05 and n=129 exist to
# attribute the bias to finite Mach or to resolution respectively.
# NOTE: groups are appended, never reordered -- the cluster shard command is
# ``--index i/len(case_manifest())`` and frozen at submit time, so inserting a
# group in the middle would silently re-assign shards.  ``(33, 0.1)`` is the
# coarse *pilot* group: it is the only ladder cheap enough to run on a loaded
# control node, and it is reported as a pilot (larger h => larger discretisation
# floor), never as the production-resolution answer.
GROUPS: tuple[tuple[int, float], ...] = ((65, 0.1), (65, 0.05), (129, 0.1), (33, 0.1))
RE_LADDER: dict[tuple[int, float], tuple[float, ...]] = {
    (65, 0.1): (40.0, 60.0, 80.0, 100.0, 120.0, 150.0),
    (65, 0.05): (60.0, 80.0, 100.0, 120.0),
    (129, 0.1): (60.0, 80.0, 100.0),
    (33, 0.1): (60.0, 80.0, 100.0, 130.0),
}


# --- node -> physical coordinate mappings ----------------------------------

def _map_node(j: float, n: int) -> float:
    return j / (n - 1)


def _map_half(j: float, n: int) -> float:
    return (j + 0.5) / n


def _map_halfbb(j: float, n: int) -> float:
    return (j - 0.5) / (n - 1.5)


MAPPINGS: dict[str, Callable[[float, int], float]] = {
    "node": _map_node,
    "half": _map_half,
    "halfbb": _map_halfbb,
}
CONV_TAGS: tuple[str, ...] = tuple(MAPPINGS)


# ---------------------------------------------------------------------------
# case manifest
# ---------------------------------------------------------------------------


def case_manifest() -> list[dict[str, Any]]:
    """Flat list of ``(n, u_lid, Re)`` cases, in a deterministic order."""
    cases: list[dict[str, Any]] = []
    for n, u_lid in GROUPS:
        for re in RE_LADDER[(n, u_lid)]:
            cases.append(
                {
                    "id": f"n{n}_u{u_lid:g}_Re{re:g}",
                    "n": n,
                    "u_lid": u_lid,
                    "re": re,
                    "tau": tau_from_reynolds(re, u_lid, n - 1),
                }
            )
    return cases


# ---------------------------------------------------------------------------
# steady roll-out
# ---------------------------------------------------------------------------


def run_steady(
    n: int,
    u_lid: float,
    re: float,
    *,
    steps_max: int = 120_000,
    check_every: int = 200,
    tol: float = 1e-7,
    use_compile: bool = True,
    log: Any = print,
) -> dict[str, Any]:
    """March the cavity to a steady state and return fields + diagnostics.

    Convergence = ``max|u_x(t) - u_x(t - check_every)| / u_lid < tol``.
    """
    geom = CavityGeometry(ny=n, nx=n, dtype=DTYPE)
    tau = torch.tensor(tau_from_reynolds(re, u_lid, n - 1), dtype=DTYPE)
    u = torch.tensor(float(u_lid), dtype=DTYPE)
    wall = geom.wall_mask
    f = initial_field(geom)

    one: Any = step
    if use_compile:
        # compile is an optimisation, not semantics: force the compile with a
        # tiny warm-up and fall back to eager if it raises (e.g. no triton).
        try:  # pragma: no cover - environment dependent
            candidate = torch.compile(step)
            warm_geom = CavityGeometry(ny=8, nx=8, dtype=DTYPE)
            with torch.no_grad():
                candidate(initial_field(warm_geom), tau, u, warm_geom.wall_mask)
            one = candidate
        except Exception as exc:  # noqa: BLE001
            log(f"[warn] torch.compile unavailable ({type(exc).__name__}: {exc}); eager fallback")
            one = step

    t0 = time.time()
    steps = 0
    residual = math.inf
    previous_ux: torch.Tensor | None = None
    status = "steps_max"
    with torch.no_grad():
        while steps < steps_max:
            for _ in range(check_every):
                f = one(f, tau, u, wall)
            steps += check_every
            _, ux, uy = macroscopic(f)
            if not torch.isfinite(ux).all() or not torch.isfinite(uy).all():
                return {
                    "status": "diverged",
                    "steps": steps,
                    "wall_seconds": time.time() - t0,
                    "residual": None,
                }
            if previous_ux is not None:
                residual = float((ux - previous_ux).abs().max()) / u_lid
                if residual < tol:
                    status = "converged"
                    break
            previous_ux = ux.clone()

    _, ux, uy = macroscopic(f)
    return {
        "status": status,
        "steps": steps,
        "residual": residual,
        "wall_seconds": time.time() - t0,
        "ux": ux.double(),
        "uy": uy.double(),
    }


# ---------------------------------------------------------------------------
# external loss against the Ghia tables
# ---------------------------------------------------------------------------


def ghia_loss(
    ux: torch.Tensor,
    uy: torch.Tensor,
    u_lid: float,
    mapping: str = "node",
    *,
    interior_only: bool = True,
) -> dict[str, float]:
    """RMSE of the two centrelines against Ghia Table I under one mapping.

    ``interior_only=True`` keeps the reference rows with ``0 < y < 1`` /
    ``0 < x < 1`` only, so wall rows (which a mapping can push outside the node
    span, where ``np.interp`` clamps them) never enter the number.
    """
    import numpy as np

    if mapping not in MAPPINGS:
        msg = f"unknown mapping {mapping!r} (have {sorted(MAPPINGS)})"
        raise ValueError(msg)
    f_map = MAPPINGS[mapping]
    ny, nx = ux.shape
    if ny != nx:
        msg = f"cavity fields must be square, got {tuple(ux.shape)}"
        raise ValueError(msg)
    n = ny

    y_nodes = np.array([f_map(j, n) for j in range(n)], dtype=float)
    x_nodes = y_nodes

    u_prof = ux.detach().cpu().numpy()[:, n // 2] / u_lid
    v_prof = uy.detach().cpu().numpy()[n // 2, :] / u_lid

    ref_y = np.asarray(GHIA_RE100["y"], dtype=float)
    ref_u = np.asarray(GHIA_RE100["u"], dtype=float)
    ref_x = np.asarray(GHIA_RE100["x"], dtype=float)
    ref_v = np.asarray(GHIA_RE100["v"], dtype=float)

    u_at = np.interp(ref_y, y_nodes, u_prof)
    v_at = np.interp(ref_x, x_nodes, v_prof)
    keep_u = (ref_y > 0.0) & (ref_y < 1.0) if interior_only else np.ones_like(ref_u, bool)
    keep_v = (ref_x > 0.0) & (ref_x < 1.0) if interior_only else np.ones_like(ref_v, bool)

    rmse_u = float(np.sqrt(np.mean((u_at[keep_u] - ref_u[keep_u]) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_at[keep_v] - ref_v[keep_v]) ** 2)))
    return {
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "J": rmse_u**2 + rmse_v**2,
        "mapping": mapping,
        "interior_only": interior_only,
        "n_rows_u": int(keep_u.sum()),
        "n_rows_v": int(keep_v.sum()),
    }


def _profiles(ux: torch.Tensor, uy: torch.Tensor, u_lid: float) -> dict[str, list[float]]:
    """Normalised centrelines on the raw node grid (for the figures)."""
    xn = ux.detach().cpu().numpy() / u_lid
    yn = uy.detach().cpu().numpy() / u_lid
    n = ux.shape[0]
    return {
        "grid": [j / (n - 1) for j in range(n)],
        "u_vertical": [float(v) for v in xn[:, n // 2]],
        "v_horizontal": [float(v) for v in yn[n // 2, :]],
    }


def _vortex_centre(ux: torch.Tensor, n: int) -> dict[str, float]:
    """Primary-vortex centre from ``psi = int u_x dy`` (node-normalised).

    Reference (Ghia 1982, Re=100): ``(0.6172, 0.7344)``.
    """
    import numpy as np

    psi = np.cumsum(ux.detach().cpu().numpy(), axis=0)
    interior = psi[1:-1, 1:-1]
    iy, ix = np.unravel_index(int(np.argmin(interior)), interior.shape)
    iy += 1
    ix += 1
    return {"x": float(ix / (n - 1)), "y": float(iy / (n - 1)),
            "psi_min": float(psi[iy, ix])}


def loss_bundle(ux: torch.Tensor, uy: torch.Tensor, u_lid: float) -> dict[str, Any]:
    """Ghia loss under every mapping (+ legacy compare_ghia + diagnostics)."""
    out: dict[str, Any] = {"conventions": {}, "legacy": {}}
    for tag in CONV_TAGS:
        out["conventions"][tag] = ghia_loss(ux, uy, u_lid, tag, interior_only=True)
    for tag, kw in (("w0_allrows", dict(wall_offset=0.0)),
                    ("w05_internal", dict(wall_offset=0.5, exclude_walls=True))):
        r = compare_ghia(ux, uy, u_lid, GHIA_RE100, **kw)
        out["legacy"][tag] = {"rmse_u": float(r["rmse_u"]), "rmse_v": float(r["rmse_v"])}
    out["profiles"] = _profiles(ux, uy, u_lid)
    out["vortex"] = _vortex_centre(ux, ux.shape[0])
    return out


# ---------------------------------------------------------------------------
# shard driver
# ---------------------------------------------------------------------------


def _parse_index(text: str) -> tuple[int, int]:
    try:
        i_s, n_s = text.split("/")
        i, total = int(i_s), int(n_s)
    except Exception as exc:  # noqa: BLE001
        msg = f"--index must look like '3/13', got {text!r}"
        raise SystemExit(msg) from exc
    if total < 1 or not 1 <= i <= total:
        msg = f"--index out of range: {i}/{total}"
        raise SystemExit(msg)
    return i, total


def run_shard(index: str, outdir: Path, steps_max: int, tol: float,
              use_compile: bool) -> dict[str, Any]:
    i, total = _parse_index(index)
    case = case_manifest()[i - 1]
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{case['id']}.json"
    started = time.time()
    payload: dict[str, Any] = {**case, "index": f"{i}/{total}"}
    steady = run_steady(case["n"], case["u_lid"], case["re"],
                        steps_max=steps_max, tol=tol, use_compile=use_compile)
    payload["run"] = {k: v for k, v in steady.items() if k not in ("ux", "uy")}
    if steady["status"] == "diverged":
        payload["loss"] = None
        payload["case_status"] = "diverged"
    else:
        payload["loss"] = loss_bundle(steady["ux"], steady["uy"], case["u_lid"])
        payload["case_status"] = "ok"
    payload["elapsed_seconds"] = time.time() - started
    path.write_text(json.dumps(payload, indent=2))
    print(f"[shard {i}/{total}] {case['id']} -> {path.name} "
          f"({payload['case_status']}, {payload['elapsed_seconds']:.0f}s)", flush=True)
    return payload


# ---------------------------------------------------------------------------
# collect: quadratic surrogate of J(Re) -> anchor Reynolds number
# ---------------------------------------------------------------------------


def quadratic_vertex(points: Sequence[tuple[float, float]]) -> dict[str, float] | None:
    """Least-squares parabola through ``points`` -> vertex.

    ``None`` for a degenerate (non-positive curvature) fit instead of a bogus
    minimum.
    """
    import numpy as np

    if len(points) < 3:
        return None
    x = np.asarray([p[0] for p in points], dtype=float)
    y = np.asarray([p[1] for p in points], dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    if not math.isfinite(a) or a <= 0.0:
        return None
    xv = float(-b / (2.0 * a))
    return {"x_min": xv, "J_min": float(a * xv * xv + b * xv + c),
            "a": float(a), "b": float(b), "c": float(c)}


def three_point_vertex(points: Sequence[tuple[float, float]]) -> dict[str, float] | None:
    """Parabola through the discrete minimum and its two ``Re`` neighbours."""
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < 3:
        return None
    k = min(range(len(pts)), key=lambda j: pts[j][1])
    if k == 0 or k == len(pts) - 1:
        return None
    (x0, y0), (x1, y1), (x2, y2) = pts[k - 1], pts[k], pts[k + 1]
    a = (y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2))
         + y2 / ((x2 - x0) * (x2 - x1)))
    b = -(y0 * (x1 + x2) / ((x0 - x1) * (x0 - x2))
          + y1 * (x0 + x2) / ((x1 - x0) * (x1 - x2))
          + y2 * (x0 + x1) / ((x2 - x0) * (x2 - x1)))
    if not math.isfinite(a) or a <= 0.0:
        return None
    return {"x_min": float(-b / (2.0 * a)), "a": float(a), "b": float(b)}


def collect(outdir: Path) -> dict[str, Any]:
    """Read every shard JSON, fit ``J(Re)`` per group, write ``summary.json``."""
    by_id = {c["id"]: c for c in case_manifest()}
    records: list[dict[str, Any]] = []
    for path in sorted(outdir.glob("n*_u*_Re*.json")):
        data = json.loads(path.read_text())
        if data.get("case_status") != "ok" or data.get("loss") is None:
            records.append({"id": data["id"], "case_status": data.get("case_status")})
            continue
        records.append(
            {
                "id": data["id"], "n": data["n"], "u_lid": data["u_lid"],
                "re": data["re"], "tau": data["tau"],
                "steps": data["run"]["steps"], "run_status": data["run"]["status"],
                "residual": data["run"]["residual"],
                "elapsed_seconds": data["elapsed_seconds"],
                "vortex": data["loss"]["vortex"],
                "profiles": data["loss"]["profiles"],
                "conv": data["loss"]["conventions"],
                "legacy": data["loss"]["legacy"],
            }
        )

    groups: dict[str, Any] = {}
    for n, u_lid in GROUPS:
        key = f"n{n}_u{u_lid:g}"
        members = [r for r in records if r.get("n") == n and r.get("u_lid") == u_lid
                   and "conv" in r]
        if not members:
            continue
        members.sort(key=lambda r: r["re"])
        points = [
            {
                "re": m["re"], "tau": m["tau"],
                **{f"{tag}_J": m["conv"][tag]["J"] for tag in CONV_TAGS},
                **{f"{tag}_rmse_u": m["conv"][tag]["rmse_u"] for tag in CONV_TAGS},
                **{f"{tag}_rmse_v": m["conv"][tag]["rmse_v"] for tag in CONV_TAGS},
            }
            for m in members
        ]
        g: dict[str, Any] = {
            "n": n, "u_lid": u_lid, "mach": u_lid * math.sqrt(3.0),
            "tau_true": tau_from_reynolds(RE_REF, u_lid, n - 1),
            "points": points, "fits": {}, "verdict": {},
        }
        for tag in CONV_TAGS:
            pts = [(p["re"], p[f"{tag}_J"]) for p in points]
            g["fits"][tag] = {
                "lsq": quadratic_vertex(pts),
                "three_point": three_point_vertex(pts),
                "discrete_min": min(pts, key=lambda p: p[1]),
            }
            fit = g["fits"][tag]["lsq"]
            if fit is None:
                g["verdict"][tag] = {"status": "no_curvature"}
                continue
            tau_anchor = tau_from_reynolds(fit["x_min"], u_lid, n - 1)
            rel = abs(tau_anchor - g["tau_true"]) / g["tau_true"]
            g["verdict"][tag] = {
                "re_anchor": fit["x_min"],
                "tau_anchor": tau_anchor,
                "tau_rel_error": rel,
                "criterion_lt_2pct": bool(rel < 0.02),
                "J_min": fit["J_min"],
                "J_at_re100": next(p[f"{tag}_J"] for p in points if p["re"] == RE_REF),
                "J_ratio_min_over_re100": fit["J_min"]
                / next(p[f"{tag}_J"] for p in points if p["re"] == RE_REF),
            }
        g["records"] = members
        groups[key] = g

    summary = {
        "anchor": "GAP-PP-2-lid-wall-diff",
        "reference": "Ghia, Ghia & Shin 1982 JCP 48:387-411, Table I (Re=100)",
        "re_ref": RE_REF,
        "loss": "J = rmse_u^2 + rmse_v^2, both normalised by u_lid, interior rows only",
        "mappings": {
            "node": "y_j = j/(n-1)",
            "half": "y_j = (j+0.5)/n",
            "halfbb": "y_j = (j-0.5)/(n-1.5)",
        },
        "cases": len(by_id),
        "groups": groups,
        "records": records,
    }
    path = outdir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"[collect] -> {path} ({len(records)} records, {len(groups)} groups)", flush=True)
    return summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def figures(outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads((outdir / "summary.json").read_text())
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    colors = {"node": "C0", "half": "C1", "halfbb": "C2"}

    groups = summary["groups"]
    if not groups:
        print("[figures] no group in summary; nothing to draw", flush=True)
        return []
    fig, axes = plt.subplots(1, len(groups), figsize=(4.8 * len(groups), 4.0), squeeze=False)
    for ax, (key, g) in zip(axes[0], groups.items()):
        pts = g["points"]
        res = [p["re"] for p in pts]
        for tag in CONV_TAGS:
            ax.plot(res, [p[f"{tag}_J"] for p in pts], "o-", ms=3.5, lw=1.1,
                    color=colors[tag], label=f"{tag}: J total")
            ax.plot(res, [p[f"{tag}_rmse_u"] ** 2 for p in pts], ":", lw=1,
                    color=colors[tag], alpha=0.7)
            ax.plot(res, [p[f"{tag}_rmse_v"] ** 2 for p in pts], "-.", lw=1,
                    color=colors[tag], alpha=0.5)
        v = g["verdict"].get("halfbb", {})
        ax.axvline(RE_REF, color="k", lw=1, ls=":")
        if "re_anchor" in v:
            ax.axvline(v["re_anchor"], color="C3", lw=1.3, ls="--")
            ax.set_title(
                f"n={g['n']}, u_lid={g['u_lid']:g} (Ma={g['mach']:.3f})\n"
                f"Re_anchor={v['re_anchor']:.1f} (tau err {100 * v['tau_rel_error']:.2f}%)",
                fontsize=9)
        ax.set_xlabel("Re = u_lid H / nu")
        ax.set_ylabel("loss component  (rmse^2)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Ghia(1982) Re=100 external anchor: loss vs model Reynolds number "
                 "(dotted line = Re=100 truth, dashed = fitted anchor)", fontsize=10)
    fig.tight_layout()
    p = figdir / "ghia_anchor_loss_vs_re.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # tau bias per group (mapping-convention spread + Ma/resolution trend)
    labels, machs, series = [], [], {tag: [] for tag in CONV_TAGS}
    for key, g in groups.items():
        labels.append(f"n={g['n']}\nu={g['u_lid']:g}")
        machs.append(g["mach"])
        for tag in CONV_TAGS:
            v = g["verdict"].get(tag, {})
            series[tag].append(100.0 * v["tau_rel_error"] if "re_anchor" in v else float("nan"))
    x = np.arange(len(labels))
    width = 0.8 / len(CONV_TAGS)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for k, tag in enumerate(CONV_TAGS):
        ax.bar(x + (k - (len(CONV_TAGS) - 1) / 2) * width, series[tag], width,
               color=colors[tag], label=f"mapping {tag}")
    ax.axhline(2.0, color="C3", ls="--", label="criterion 1: 2 %")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{lab}\nMa={m:.3f}" for lab, m in zip(labels, machs)], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("|tau_anchor - tau_true| / tau_true  [%]")
    ax.set_title("External-anchor tau bias from Ghia data alone", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = figdir / "ghia_anchor_tau_bias.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(str(p))

    # centrelines: model at tau_true vs at the anchor, main group
    main = summary["groups"].get("n65_u0.1")
    if main:
        recs = {r["re"]: r for r in main["records"]}
        v = main["verdict"]["halfbb"]
        anchor = v["re_anchor"]
        nearest = min(recs, key=lambda r: abs(r - anchor))
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9))
        ref_y = np.asarray(GHIA_RE100["y"])
        ref_x = np.asarray(GHIA_RE100["x"])
        axes[0].plot(np.asarray(GHIA_RE100["u"]), ref_y, "ko", ms=3.5, label="Ghia 1982 (Re=100)")
        axes[1].plot(ref_x, np.asarray(GHIA_RE100["v"]), "ko", ms=3.5, label="Ghia 1982 (Re=100)")
        n = main["n"]
        grid = np.arange(n) / (n - 1)
        for re_val, color, tag in ((RE_REF, "C0", "model, Re=100 (tau_true)"),
                                   (nearest, "C3", f"model, Re={nearest:g} (near anchor)")):
            if re_val not in recs:
                continue
            axes[0].plot(recs[re_val]["profiles"]["u_vertical"], grid, color=color, lw=1.2,
                         label=tag)
            axes[1].plot(grid, recs[re_val]["profiles"]["v_horizontal"], color=color, lw=1.2,
                         label=tag)
        axes[0].set_xlabel("u_x / u_lid  (vertical centreline)")
        axes[0].set_ylabel("y / H")
        axes[1].set_xlabel("x / H  (horizontal centreline)")
        axes[1].set_ylabel("u_y / u_lid")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("n=65, u_lid=0.1: model at tau_true vs tau_anchor vs Ghia", fontsize=10)
        fig.tight_layout()
        p = figdir / "ghia_anchor_profiles.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(str(p))

    print("[figures] " + ", ".join(written), flush=True)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ghia external-anchor tau inversion")
    ap.add_argument("command", choices=["shard", "collect", "figures", "manifest", "all"])
    ap.add_argument("--index", default="1/1", help="shard index 'i/N' (1-based)")
    ap.add_argument("--outdir", default="artifacts/gap_pp2_ghia_anchor")
    ap.add_argument("--steps-max", type=int, default=150_000)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8,
                    help="torch intra-op threads (tiny grids thrash above ~8)")
    args = ap.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    outdir = Path(args.outdir)
    if args.command == "manifest":
        print(json.dumps(case_manifest(), indent=2))
        return 0
    if args.command == "shard":
        run_shard(args.index, outdir, args.steps_max, args.tol, bool(args.compile))
        return 0
    if args.command == "collect":
        collect(outdir)
        return 0
    if args.command == "figures":
        figures(outdir)
        return 0
    total = len(case_manifest())
    for i in range(1, total + 1):
        run_shard(f"{i}/{total}", outdir, args.steps_max, args.tol, bool(args.compile))
    collect(outdir)
    figures(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())