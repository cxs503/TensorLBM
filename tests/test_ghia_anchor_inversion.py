"""Contract tests for the Ghia external-anchor inversion study (GAP-PP-2).

The study (``scripts/verify_gap_pp2_ghia_anchor_inversion.py``) answers a
question the same-chain acceptance test cannot: what ``tau`` does the model
prefer when the data are the *external* Ghia (1982) tables?  These tests pin
the pieces whose silent breakage would invalidate that answer:

1. ``ghia_loss`` is zero on a field that reproduces the reference table and
   moves by exactly the injected offset (no metric/sign slip);
2. the node -> physical coordinate mapping matters (the three conventions are
   distinct on the same field) and only interior reference rows enter the
   number (wall rows are excluded, count pinned);
3. the quadratic surrogates recover a known vertex and refuse to invent a
   minimum at the bracketing boundary;
4. ``collect`` turns shard JSONs into the anchor verdict with the right sign
   and the right criterion flag (synthetic parabola, no solver);
5. the case manifest covers the designed (n, u_lid) groups with ``tau`` in the
   stable range;
6. (slow) the steady roll-out + loss bundle is finite end to end.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.verify_gap_pp2_ghia_anchor_inversion as study
from tensorlbm.lid_driven_cavity import GHIA_RE100


def _sync_field(n: int, u_lid: float = 0.1, offset: float = 0.0) -> tuple:
    """Fields whose centrelines reproduce Ghia Table I exactly at the nodes."""
    y_sorted = np.asarray(GHIA_RE100["y"])[::-1]
    u_sorted = np.asarray(GHIA_RE100["u"])[::-1]
    x_nodes = np.arange(n) / (n - 1)
    u_prof = np.interp(x_nodes, y_sorted, u_sorted) + offset
    v_prof = np.interp(x_nodes, np.asarray(GHIA_RE100["x"]), np.asarray(GHIA_RE100["v"]))
    ux = torch.zeros((n, n), dtype=torch.float64)
    uy = torch.zeros((n, n), dtype=torch.float64)
    ux[:, n // 2] = torch.as_tensor(u_lid * u_prof)
    uy[n // 2, :] = torch.as_tensor(u_lid * v_prof)
    return ux, uy


def test_ghia_loss_floor_is_metric_only_and_converges() -> None:
    """A field that *is* Ghia Table I reproduces it up to the kink floor only.

    The table is piecewise linear with kinks at its own abscissae; sampling it
    on the node grid and interpolating back costs O(h * slope jump), i.e. the
    metric has a non-zero floor with no solver involved.  The floor must
    converge with n (that is what makes it separable from solver error) and
    must stay an order of magnitude below the 3 % acceptance scale.
    """
    floors = {}
    for n in (33, 129):
        ux, uy = _sync_field(n)
        out = study.ghia_loss(ux, uy, 0.1, "node")
        floors[n] = out
        assert out["rmse_u"] < 1e-2, out  # only the kink floor, no solver
        assert out["rmse_v"] < 1e-2, out
    assert floors[129]["rmse_u"] < 0.6 * floors[33]["rmse_u"], floors


def test_ghia_loss_matches_compare_ghia_on_the_legacy_convention() -> None:
    """Independent re-implementation must agree with the production metric."""
    from tensorlbm.lid_driven_cavity import compare_ghia

    n = 33
    ux, uy = _sync_field(n)
    ux = ux + torch.linspace(0.0, 0.02, n, dtype=torch.float64).view(-1, 1)
    mine = study.ghia_loss(ux, uy, 0.1, "node", interior_only=False)
    theirs = compare_ghia(ux, uy, 0.1, GHIA_RE100)  # wall_offset=0.0, all rows
    assert mine["rmse_u"] == pytest.approx(theirs["rmse_u"], rel=1e-12)
    assert mine["rmse_v"] == pytest.approx(theirs["rmse_v"], rel=1e-12)


def test_ghia_loss_reports_the_injected_offset() -> None:
    n = 33
    base = study.ghia_loss(*_sync_field(n), 0.1, "node")
    ux, uy = _sync_field(n, offset=0.05)
    shifted = study.ghia_loss(ux, uy, 0.1, "node")
    assert shifted["rmse_u"] > base["rmse_u"]  # a wrong profile must cost more
    assert shifted["rmse_u"] == pytest.approx(0.05, abs=1e-2)
    assert shifted["rmse_v"] == pytest.approx(base["rmse_v"], rel=1e-12)  # u-only offset


def test_mapping_changes_the_number_and_wall_rows_are_excluded() -> None:
    n = 33
    ux, uy = _sync_field(n)
    # a field that is *not* a perfect solution: the mapping must matter
    ux = ux + torch.linspace(0.0, 0.01, n, dtype=torch.float64).view(-1, 1)
    losses = {tag: study.ghia_loss(ux, uy, 0.1, tag) for tag in study.CONV_TAGS}
    assert set(losses) == {"node", "half", "halfbb"}
    values = [losses[tag]["rmse_u"] for tag in study.CONV_TAGS]
    assert len(set(round(v, 12) for v in values)) == 3, values
    for tag, out in losses.items():
        assert out["n_rows_u"] == len(GHIA_RE100["y"]) - 2, tag
        assert out["n_rows_v"] == len(GHIA_RE100["x"]) - 2, tag
        assert math.isfinite(out["J"])


def test_quadratic_vertex_recovers_a_known_minimum() -> None:
    points = [(float(re), 1e-6 * (re - 80.0) ** 2) for re in (40.0, 60.0, 80.0, 100.0)]
    lsq = study.quadratic_vertex(points)
    assert lsq is not None
    assert lsq["x_min"] == pytest.approx(80.0, abs=1e-9)
    three = study.three_point_vertex(points)
    assert three is not None
    assert three["x_min"] == pytest.approx(80.0, abs=1e-9)


def test_vertex_is_not_invented_on_a_boundary_minimum() -> None:
    points = [(40.0, 1e-6), (60.0, 2e-6), (80.0, 4e-6)]
    assert study.three_point_vertex(points) is None
    assert study.quadratic_vertex(points[:2]) is None  # fewer than 3 points


def _fake_shard(path: Path, n: int, u_lid: float, re: float, J: float) -> None:
    """Minimal shard JSON in the exact shape ``collect`` consumes."""
    conv = {
        tag: {"rmse_u": math.sqrt(J), "rmse_v": 1e-3, "J": J + 1e-6,
              "mapping": tag, "interior_only": True, "n_rows_u": 15, "n_rows_v": 15}
        for tag in study.CONV_TAGS
    }
    payload = {
        "id": f"n{n}_u{u_lid:g}_Re{re:g}", "n": n, "u_lid": u_lid, "re": re,
        "tau": 0.5 + 3.0 * u_lid * (n - 1) / re,
        "run": {"status": "converged", "steps": 1000, "residual": 1e-9,
                "wall_seconds": 1.0},
        "loss": {"conventions": conv,
                 "legacy": {"w0_allrows": {"rmse_u": 1.0, "rmse_v": 1.0}},
                 "vortex": {"x": 0.5, "y": 0.5, "psi_min": -0.1},
                 "profiles": {"grid": [0.0, 1.0], "u_vertical": [0.0, 0.0],
                              "v_horizontal": [0.0, 0.0]}},
        "case_status": "ok",
        "elapsed_seconds": 1.0,
    }
    path.write_text(json.dumps(payload))


def test_collect_produces_the_anchor_verdict(tmp_path: Path) -> None:
    # synthetic parabola in Re with its minimum at Re = 70 (biased away from 100)
    for re in (60.0, 80.0, 100.0, 120.0, 150.0):
        _fake_shard(tmp_path / f"n65_u0.1_Re{re:g}.json", 65, 0.1, re,
                    3.0e-5 + 1e-6 * (re - 70.0) ** 2)
    summary = study.collect(tmp_path)
    group = summary["groups"]["n65_u0.1"]
    verdict = group["verdict"]["halfbb"]
    assert verdict["re_anchor"] == pytest.approx(70.0, abs=0.5)
    tau_true = 0.5 + 3.0 * 0.1 * 64.0 / 100.0
    assert verdict["tau_anchor"] == pytest.approx(0.5 + 3.0 * 0.1 * 64.0 / 70.0, rel=1e-3)
    assert verdict["tau_rel_error"] == pytest.approx(
        abs(tau_true - verdict["tau_anchor"]) / tau_true, rel=1e-9
    )
    assert verdict["criterion_lt_2pct"] is False  # a 30 % Re bias is not < 2 % in tau
    assert group["tau_true"] == pytest.approx(tau_true)
    assert group["mach"] == pytest.approx(0.1 * math.sqrt(3.0))


def test_case_manifest_covers_the_designed_groups() -> None:
    cases = study.case_manifest()
    assert len(cases) == sum(len(v) for v in study.RE_LADDER.values()) == 13
    groups = {(c["n"], c["u_lid"]) for c in cases}
    assert groups == {(65, 0.1), (65, 0.05), (129, 0.1)}
    assert any(c["re"] == study.RE_REF for c in cases)  # the truth Resolution is exercised
    for c in cases:  # BGK stability floor: tau must stay clear of 0.5
        assert 0.55 < c["tau"] < 1.2, c
        assert c["tau"] == pytest.approx(0.5 + 3.0 * c["u_lid"] * (c["n"] - 1) / c["re"])


@pytest.mark.slow
def test_steady_rollout_and_loss_bundle_are_finite() -> None:
    # NB: on a 17-node cavity Re=100 would need tau = 0.548, inside the BGK
    # stability floor (see docs/GAP_PP_2_LID_WALL_DIFF.md §8.4); use Re=50 so
    # that tau = 0.596.  This is why the production ladders start at n=65.
    state = study.run_steady(17, 0.1, 50.0, steps_max=800, check_every=200,
                             tol=1e-12, use_compile=False, log=lambda *_: None)
    assert state["status"] in {"converged", "steps_max"}
    bundle = study.loss_bundle(state["ux"], state["uy"], 0.1)
    for tag in study.CONV_TAGS:
        assert math.isfinite(bundle["conventions"][tag]["J"])
    assert math.isfinite(bundle["vortex"]["x"])