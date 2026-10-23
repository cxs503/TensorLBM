"""B4-P3b · acquisition-step latency + oracle-match bench.

Times :func:`propose_acquisition` per strategy on the B4-P3b demo inputs
(flagged queries on the B4-fam hull-form families, v4 corpus guard
features) and reports the oracle-match rate of each proposal set against
``cache_fam.npz``.  Falls back to a fully synthetic corpus (geometry
frontend only, no /nfs artifacts) when the run data is absent, so the
benchmark stays runnable anywhere.  ``max_disagreement`` additionally
exercises the served 5-member ensemble std when the b4_serve checkpoints
exist.

Usage::

    PYTHONPATH=src python benchmarks/b4_al_acquisition_bench.py \
        [--budget 16] [--candidates 512] [--out-dir DIR]

Writes ``al_acquisition_bench.json`` into ``--out-dir`` (default: the run
dir ``/nfs/wangxi/runs/b4_al_20260825`` when present, else cwd).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import tensorlbm
from tensorlbm.ai.active_learning import (
    HULLFORM_AXES,
    FlaggedQuery,
    ServiceSpec,
    corpus_cond_v3,
    corpus_design_keys,
    hullform_condition_rows,
    labels_from_cache,
    load_corpus_index,
    predict_design,
    propose_acquisition,
)
from tensorlbm.ai.drag_cond import PRODUCTION_GRID

V4 = "/nfs/wangxi/runs/b4_v4_20260824/cache_v4.npz"
FAM = "/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz"
FAM_META = "/nfs/wangxi/runs/b4_fam_20260824/cache_fam_meta.json"
SERVE = Path("/nfs/wangxi/runs/b4_serve_20260824/ckpts")
RUN_DIR = Path("/nfs/wangxi/runs/b4_al_20260825")
STRATEGIES = ("envelope_shell", "max_disagreement", "coverage")


def synthetic_inputs(n: int = 24, seed: int = 11) -> tuple[list[FlaggedQuery], np.ndarray]:
    """Flagged queries + a mother-like condition cloud, no /nfs needed."""
    rng = np.random.default_rng(seed)
    queries: list[FlaggedQuery] = []
    mother_params: dict[str, Any] = {
        "hull_type": "with_sail",
        "sail_scale": 1.0,
        "fin_scale": 1.0,
        "u_in": 0.10,
    }
    for i, re in enumerate(np.geomspace(65.0, 680.0, n)):
        params = dict(mother_params)
        for a in HULLFORM_AXES:
            params[a] = float(rng.choice([0.75, 1.0, 1.3]))
        queries.append(
            FlaggedQuery(
                params=params,
                re=float(re),
                verdict="review",
                score=float(3.0 + 3.0 * rng.random()),
                member_std=float(0.02 + 0.05 * rng.random()),
            )
        )
    cond = hullform_condition_rows(
        mother_params, list(np.geomspace(65.0, 680.0, 20)), grid=PRODUCTION_GRID
    )
    return queries, cond


def demo_flagged_queries(corpus: Any, existing_cond: np.ndarray) -> list[FlaggedQuery]:
    """The B4-P3b demo flagged queries: B4-fam rows 1-2 of each family
    (stride-4 split), guard-scored by the v4-corpus channel guard."""
    from tensorlbm.ai.active_learning import honest_verdict
    from tensorlbm.ai.inference_service import EnvelopeMahalanobisGuardrail

    z = np.load(FAM)
    meta = json.loads(Path(FAM_META).read_text())
    rows = np.nonzero(np.asarray(z["dsi"]) >= 6)[0]
    fam_rows: dict[str, list[dict[str, Any]]] = {}
    for i, m in enumerate(meta):
        m = dict(m)
        m["_cd"] = float(z["cd"][int(rows[i])])
        fam_rows.setdefault(str(m["fam"]), []).append(m)
    guard = EnvelopeMahalanobisGuardrail(existing_cond)
    mother_env = {a: (1.0, 1.0) for a in HULLFORM_AXES}
    queries: list[FlaggedQuery] = []
    for fam in sorted(fam_rows):
        rs = sorted(fam_rows[fam], key=lambda m: float(m["re"]))
        for m in rs[1::4] + rs[2::4]:
            params: dict[str, Any] = {
                "hull_type": str(m["hull"]),
                "sail_scale": float(m["sail"]),
                "fin_scale": float(m["fin"]),
                "u_in": float(m["u_in"]),
            }
            params.update({a: float(m[a]) for a in HULLFORM_AXES})
            re = float(m["re"])
            c = hullform_condition_rows(params, [re], grid=PRODUCTION_GRID)
            chan = guard.check(c)
            queries.append(
                FlaggedQuery(
                    params=params,
                    re=re,
                    verdict=honest_verdict(chan, params, mother_env).flag,
                    score=float(chan.score),
                    member_std=0.03,
                )
            )
    return queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=512)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    have_real = Path(V4).exists() and Path(FAM).exists()
    out_dir = args.out_dir or (RUN_DIR if RUN_DIR.exists() else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: list[FlaggedQuery]
    if have_real:
        corpus = load_corpus_index(V4)
        existing_cond = corpus_cond_v3(corpus)
        queries = demo_flagged_queries(corpus, existing_cond)
    else:
        corpus = None
        queries, existing_cond = synthetic_inputs()

    member_std_fn = None
    if have_real and SERVE.exists():
        import torch

        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        spec = ServiceSpec(
            ckpt_dir=SERVE,
            guard_features=existing_cond,
            axes_env={a: (1.0, 1.0) for a in HULLFORM_AXES},
            corpus_cache=corpus["x"],
            cache_re=corpus["re"],
            cache_designs=corpus_design_keys(corpus),
            device=dev,
        )
        backend = spec.backend()

        def std_fn(pts: list[Any]) -> np.ndarray:
            out = []
            for p in pts:
                _m, _mean, s = predict_design(backend, spec, p.params, [p.re], grid=PRODUCTION_GRID)
                out.append(float(s[0]))
            return np.asarray(out)

        member_std_fn = std_fn
    if member_std_fn is None:

        def pseudo_std_fn(pts: list[Any]) -> np.ndarray:
            # deterministic stand-in when no served ensemble exists
            vals = []
            for p in pts:
                seedv = p.re + sum(float(p.params.get(a, 1.0)) for a in HULLFORM_AXES)
                vals.append(0.01 + 0.05 * abs(np.sin(seedv)))
            return np.asarray(vals)

        member_std_fn = pseudo_std_fn

    rows: list[dict[str, Any]] = []
    for strat in STRATEGIES:
        t0 = time.perf_counter()
        pts = propose_acquisition(
            queries,
            strategy=strat,
            budget=args.budget,
            existing_cond=existing_cond,
            grid=PRODUCTION_GRID,
            seed=20260825,
            n_candidates=args.candidates,
            member_std_fn=member_std_fn if strat == "max_disagreement" else None,
        )
        dt = time.perf_counter() - t0
        row: dict[str, Any] = {
            "strategy": strat,
            "n_proposed": len(pts),
            "latency_s": round(dt, 3),
            "source": "real" if have_real else "synthetic",
        }
        if have_real:
            labs = labels_from_cache(pts, FAM)
            row["n_oracle_matched"] = int(sum(l.matched for l in labs))
        rows.append(row)
        extra = f" | oracle-matched {row['n_oracle_matched']}" if have_real else ""
        print(f"{strat:18s} proposed {len(pts):2d} in {dt:6.2f}s{extra}")

    out = out_dir / "al_acquisition_bench.json"
    out.write_text(
        json.dumps(
            {
                "tensorlbm": tensorlbm.__file__,
                "budget": args.budget,
                "n_candidates": args.candidates,
                "strategies": rows,
            },
            indent=1,
        )
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
