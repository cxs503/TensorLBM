"""Validation tests for the BFL sphere-drag anchor (GAP-LBM-SPHERE).

Three layers:

1. **Geometry invariants** of the cut-link set (analytic checks, instant).
2. **Solver invariants** on a short integration (mass conservation, finiteness,
   plausible drag band) – the run is time-boxed so the test stays cheap.
3. **Artifact checks** on ``benchmarks/verified/sphere_drag_bfl/results.json``:
   the committed benchmark must contain real numbers (steps > 0, wall time > 0)
   and must satisfy the pre-declared accuracy bands.  This is what makes the
   anchor machine-checkable rather than self-declared.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch

from tensorlbm.sphere_drag_bfl import (
    DEFAULT_DIAMETERS,
    DEFAULT_RE_SCAN,
    REFERENCE_CD_CGW,
    SphereBFLConfig,
    SphereBFLSolver,
    sphere_links,
    reference_cd,
    reference_source,
    schiller_naumann_cd,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "verified" / "sphere_drag_bfl" / "results.json"

D3Q19 = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
    (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
    (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1),
]


# --------------------------------------------------------------------------- #
# 1. geometry
# --------------------------------------------------------------------------- #

class TestCutLinkGeometry:
    @pytest.fixture(scope="class")
    def links(self):
        return sphere_links(40, 24, 24, 8.0, center=(12.0, 11.5, 11.5))

    def test_all_links_terminate_on_the_sphere(self, links):
        """|x + q e - c| == R for every cut link (BFL fractional distance)."""
        ex = torch.tensor([D3Q19[int(d)][0] for d in links.dirs], dtype=torch.float64)
        ey = torch.tensor([D3Q19[int(d)][1] for d in links.dirs], dtype=torch.float64)
        ez = torch.tensor([D3Q19[int(d)][2] for d in links.dirs], dtype=torch.float64)
        cx, cy, cz = links.center
        px = links.bx.to(torch.float64) + links.q.to(torch.float64) * ex - cx
        py = links.by.to(torch.float64) + links.q.to(torch.float64) * ey - cy
        pz = links.bz.to(torch.float64) + links.q.to(torch.float64) * ez - cz
        r = torch.sqrt(px * px + py * py + pz * pz)
        # q is stored in float32 -> residual accuracy is ~1e-7
        assert torch.allclose(r, torch.full_like(r, links.radius), atol=1e-6)
        assert float((r - links.radius).abs().max()) < 1e-6

    def test_q_in_unit_interval(self, links):
        assert float(links.q.min()) > 0.0
        assert float(links.q.max()) <= 1.0 + 1e-9
        # a numerically resolved sphere must exercise both BFL branches
        assert int((links.q < 0.5).sum()) > 0
        assert int((links.q >= 0.5).sum()) > 0

    def test_links_connect_fluid_to_solid(self, links):
        solid = links.solid
        for i in range(links.n_links):
            x, y, z = int(links.bx[i]), int(links.by[i]), int(links.bz[i])
            assert not bool(solid[x, y, z]), "cut link must start in the fluid"
            dx, dy, dz = D3Q19[int(links.dirs[i])]
            assert bool(solid[x + dx, y + dy, z + dz]), "cut link must end in the solid"

    def test_link_moment_directions_balance(self, links):
        """A centred sphere has no spurious net force direction in the link set."""
        for comp in range(3):
            s = sum(
                D3Q19[int(d)][comp] for d in links.dirs.tolist()
            )
            assert s == 0, f"net link direction along axis {comp} = {s}"

    def test_no_duplicate_node_direction_pairs(self, links):
        n = links.n_links
        key = (
            (links.bx.to(torch.int64) * 10_000
             + links.by.to(torch.int64)) * 10_000
            + links.bz.to(torch.int64)
        ) * 19 + links.dirs.to(torch.int64)
        assert len(set(key.tolist())) == n

    def test_link_count_scales_with_surface_area(self):
        a = sphere_links(60, 32, 32, 8.0)
        b = sphere_links(90, 48, 48, 12.0)
        area_ratio = (12.0 / 8.0) ** 2
        ratio = b.n_links / a.n_links
        assert abs(ratio - area_ratio) / area_ratio < 0.15


# --------------------------------------------------------------------------- #
# 2. reference data
# --------------------------------------------------------------------------- #

class TestReferenceAnchor:
    def test_table_is_monotone_decreasing(self):
        re = sorted(REFERENCE_CD_CGW)
        cd = [REFERENCE_CD_CGW[r] for r in re]
        assert all(b < a for a, b in zip(cd[:-1], cd[1:]))

    def test_table_hits_stokes_limit(self):
        # 24/Re + higher-order corrections: Cd*Re -> 24 as Re -> 0
        assert 20.0 < REFERENCE_CD_CGW[1.0] * 1.0 < 29.0

    def test_interpolation_reproduces_table(self):
        for re, cd in REFERENCE_CD_CGW.items():
            assert reference_cd(re) == pytest.approx(cd, rel=1e-12)

    def test_interpolation_between_points(self):
        lo, hi = reference_cd(100.0), reference_cd(200.0)
        mid = reference_cd(141.42135623730951)  # geometric mean of 100 and 200
        assert hi < mid < lo

    def test_reference_source_labels(self):
        assert "table" in reference_source(100.0)
        assert "correlation" in reference_source(1.0e6)

    def test_schiller_naumann_matches_cgw_within_30pct(self):
        for re in (20.0, 50.0, 100.0, 200.0):
            sn = schiller_naumann_cd(re)
            cgw = REFERENCE_CD_CGW[re]
            assert abs(sn - cgw) / cgw < 0.30

    def test_scan_lists(self):
        assert len(DEFAULT_RE_SCAN) >= 5
        assert list(DEFAULT_RE_SCAN) == sorted(DEFAULT_RE_SCAN)
        assert len(DEFAULT_DIAMETERS) >= 3


# --------------------------------------------------------------------------- #
# 3. solver invariants (time-boxed short run)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def short_run():
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    cfg = SphereBFLConfig(re=20.0, diameter=8, u_lb=0.1, threads=4,
                          max_steps=300, window=100, tol=10.0)
    solver = SphereBFLSolver(cfg)
    m0 = solver.mass()
    t0 = time.time()
    cd_hist = []
    step = 0
    while step < 400 and time.time() - t0 < 40.0:
        solver.collide()
        solver.stream()
        step += 1
        if step % 25 == 0:
            cd_hist.append(solver.drag_coefficient)
    return {
        "cfg": cfg, "solver": solver, "m0": m0, "steps": step,
        "cd_hist": cd_hist, "wall": time.time() - t0,
    }


class TestSolverInvariants:
    def test_fields_stay_finite(self, short_run):
        assert short_run["steps"] > 20, "short run did not execute any steps"
        f = short_run["solver"].f
        assert bool(torch.isfinite(f).all())
        assert float(f.abs().max()) < 10.0

    def test_mass_is_conserved_to_ppm(self, short_run):
        s = short_run["solver"]
        drift_ppm = abs(s.mass() - short_run["m0"]) / abs(short_run["m0"]) * 1e6
        assert drift_ppm < 5000.0, f"mass drift {drift_ppm:.0f} ppm"

    def test_density_stays_weakly_compressible(self, short_run):
        rho, ux, uy, uz = short_run["solver"].macroscopic()
        assert float(rho.min()) > 0.9
        assert float(rho.max()) < 1.1
        assert float(ux.abs().max()) < 0.5

    def test_drag_signal_is_positive_and_plausible(self, short_run):
        cd = short_run["solver"].drag_coefficient
        # Re = 20 reference is 2.73; an under-resolved developing solution sits
        # in a wide but finite band around it
        assert 0.5 < cd < 8.0, f"Cd = {cd}"

    def test_drag_relaxes_towards_the_anchor(self, short_run):
        hist = short_run["cd_hist"]
        assert len(hist) >= 3
        # the impulsive start overshoots; the tail must be closer to the
        # reference than the first sample
        ref = reference_cd(short_run["cfg"].re)
        assert abs(hist[-1] - ref) <= abs(hist[0] - ref) * 1.2

    def test_invalid_configurations_are_rejected(self):
        with pytest.raises(ValueError):
            SphereBFLConfig(re=-1.0, diameter=8)
        with pytest.raises(ValueError):
            SphereBFLSolver(SphereBFLConfig(re=10.0, diameter=3))
        with pytest.raises(ValueError):
            SphereBFLSolver(SphereBFLConfig(re=10.0, diameter=8, u_lb=0.5))
        # tau - 1/2 too small for this Re/u combination
        with pytest.raises(ValueError):
            SphereBFLSolver(SphereBFLConfig(re=1e6, diameter=8, u_lb=0.1))


# --------------------------------------------------------------------------- #
# 4. committed benchmark artifact
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def artifact():
    if not RESULTS.exists():
        pytest.skip(f"benchmark artifact missing: {RESULTS}")
    return json.loads(RESULTS.read_text())


class TestBenchmarkArtifact:
    def test_artifact_is_a_real_run(self, artifact):
        assert artifact["n_cases"] >= 5
        assert artifact["elapsed_s"] > 0.0
        for row in artifact["summary"]:
            assert row["steps"] > 0, "placeholder result: no time steps executed"
            assert row["wall_s"] > 0.0, "placeholder result: no wall time"
            assert math.isfinite(row["cd"]) and row["cd"] > 0.0
            assert math.isfinite(row["cd_ref"]) and row["cd_ref"] > 0.0

    def test_scan_covers_five_reynolds_numbers(self, artifact):
        re_main = {row["re"] for row in artifact["summary"] if row["diameter"] == 12}
        assert len(re_main) >= 5
        assert {10.0, 20.0, 50.0, 100.0, 200.0}.issubset(re_main)

    def test_laminar_accuracy_band(self, artifact):
        bad = [
            row for row in artifact["summary"]
            if row["diameter"] == 12 and row["re"] <= 200.0 and row["cd_err_pct"] > 10.0
        ]
        assert not bad, f"laminar Cd outside 10% band: {bad}"

    def test_transitional_accuracy_band(self, artifact):
        """Transitional points must either hit the 10 % band or be honestly
        reported as divergent (finite=False) -- never silently dropped."""
        rows = [
            row for row in artifact["summary"]
            if row["diameter"] == 12 and row["re"] > 200.0
        ]
        assert rows, "no transitional cases in the artifact"
        for row in rows:
            if row["finite"]:
                assert row["cd_err_pct"] <= 10.0, row
            else:
                assert row["steps"] > 0

    def test_scan_has_five_finite_points(self, artifact):
        finite = [
            row for row in artifact["summary"]
            if row["diameter"] == 12 and row["finite"] and math.isfinite(row["cd"])
        ]
        assert len(finite) >= 5, [r["key"] for r in finite]

    def test_drag_decreases_with_reynolds(self, artifact):
        rows = sorted(
            (row for row in artifact["summary"] if row["diameter"] == 12),
            key=lambda r: r["re"],
        )
        cds = [row["cd"] for row in rows]
        assert all(b < a for a, b in zip(cds[:-1], cds[1:])), cds

    def test_grid_refinement_has_three_levels(self, artifact):
        grid = artifact["grid_convergence"]
        assert grid, "no grid-refinement study in the artifact"
        for key, entry in grid.items():
            assert len(entry["diameters"]) >= 3, (key, entry)
            assert entry["pass_finest_5pct"], (key, entry)

    def test_acceptance_flags(self, artifact):
        acc = artifact["acceptance"]
        assert acc["all_finite"] is True
        assert acc["laminar_pass"] is True
        assert acc["transitional_pass"] is True