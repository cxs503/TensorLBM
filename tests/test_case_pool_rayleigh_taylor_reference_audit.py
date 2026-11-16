"""CASE-RAYLEIGH-TAYLOR: reference-convention and model-symmetry audit.

Why this file exists
--------------------
The case's nonlinear gate compares the saturated bubble/spike front speed with
the potential-flow reference

    u_b = sqrt(2 A g / ((1+A) C k)),   u_s = sqrt(2 A g / ((1-A) C k))

and the *whole verdict* hinges on ``C``.  Round 1 hard-coded ``C = 3``; round 2
(revision m2026-11-13) silently re-defined it locally as ``C = 1`` after
back-solving it **from the solver's own terminal velocities** (quoted in the
old comment: "C = 0.90 (A = 1/2) and C = 1.01 (A = 1/3)").  That is circular: a
reference fitted to the answer always passes.  The assertions below make the
three failure modes impossible to reintroduce silently:

1. the constant lives in exactly one place (``ref/ref_lock.py``) and no module
   re-defines it locally;
2. both literature conventions are carried, locked to the two source excerpts
   that disagree, so the verdict's convention-dependence stays visible;
3. the bubble/spike *asymmetry* gate is unattainable in this model class -- the
   single-density Boussinesq equations are exactly invariant under
   ``(phi, uy, y) -> (-phi, -uy, -y)``, which exchanges bubble and spike.  The
   invariance is asserted numerically to float32 round-off for both phase-field
   schemes, so ``u_s/u_b = 1`` is a theorem, not a measurement (round 2 reported
   ``u_s/u_b = 1.000`` as if it were evidence about the physics).

Run: PYTHONPATH=src python -m pytest -q tests/test_case_pool_rayleigh_taylor_reference_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "case_pool" / "rayleigh_taylor"
for _p in (str(REPO / "src"), str(BENCH), str(BENCH / "ref")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_case as rc                                        # noqa: E402
import ref_lock                                              # noqa: E402
from tensorlbm.d2q9 import W, equilibrium, macroscopic       # noqa: E402
from tensorlbm.rayleigh_taylor import RTConfig, rt_step      # noqa: E402

# ---------------------------------------------------------------------------
# 1. one source of truth for the geometric constant
# ---------------------------------------------------------------------------


def test_constant_is_imported_not_redefined() -> None:
    """The gate constant must *be* ``ref_lock``'s, and must not be a local literal."""
    assert rc.C_2D == ref_lock.C_2D
    assert rc.C_2D_ALT == ref_lock.C_2D_ALT
    src = (BENCH / "run_case.py").read_text()
    offenders = [ln.strip() for ln in src.splitlines()
                 if ln.lstrip().startswith("C_2D") and "=" in ln and "import" not in ln]
    assert not offenders, f"local re-definition of the reference constant: {offenders}"
    # the back-solve that produced the false PASS in round 2 must not come back
    for fitted in ("C = 0.90", "C = 1.01", "C_2D = 0.9", "C_2D = 1.01"):
        assert fitted not in src, f"back-solved constant back in the driver: {fitted}"


def test_primary_constant_matches_the_2d_primary_source() -> None:
    """C = 3 reproduces the *2-D* Fr_B quoted by arXiv:1910.03663 Fig. 3."""
    fr_c3 = math.sqrt(2.0 / (rc.C_2D * 2.0 * math.pi))
    assert rc.C_2D == 3.0
    assert fr_c3 == pytest.approx(math.sqrt(1.0 / (3.0 * math.pi)), abs=1e-15)
    # ... and the competing convention is locked the same way, not remembered
    fr_c1 = math.sqrt(2.0 / (rc.C_2D_ALT * 2.0 * math.pi))
    assert rc.C_2D_ALT == 1.0
    assert fr_c1 == pytest.approx(math.sqrt(1.0 / math.pi), abs=1e-15)
    assert fr_c1 / fr_c3 == pytest.approx(math.sqrt(3.0), rel=1e-15)


def test_convention_conflict_is_recorded_with_both_sources() -> None:
    conflict = ref_lock.CONVENTION_CONFLICT
    assert conflict["primary"]["c"] == rc.C_2D
    assert conflict["alternate"]["c"] == rc.C_2D_ALT
    assert "1910.03663" in conflict["primary"]["source"]
    assert "2009.14655" in conflict["alternate"]["source"]
    assert conflict["status"].startswith("UNRESOLVED")
    assert "no constant is fitted to solver output" in conflict["policy"]
    # the source excerpts both exist on disk, so the disagreement is auditable
    for name in ("arxiv_1910.03663_fig3_caption_p6.txt",
                 "arxiv_2009.14655_eq1_p3.txt"):
        assert (BENCH / "ref" / "sources" / name).is_file()


def test_boussinesq_reference_uses_the_locked_constant() -> None:
    """``boussinesq_velocity`` is sqrt(2Ag/(Ck)) with C from the lock, nothing else."""
    for atwood, g, lam in ((1 / 3, 2e-4, 64), (0.5, 2e-4, 128), (0.2, 1e-4, 256)):
        k = 2.0 * math.pi / lam
        assert rc.boussinesq_velocity(atwood, g, lam) == pytest.approx(
            math.sqrt(2.0 * atwood * g / (3.0 * k)), rel=1e-14)
        assert rc.boussinesq_velocity_alt(atwood, g, lam) == pytest.approx(
            math.sqrt(2.0 * atwood * g / (1.0 * k)), rel=1e-14)
        # the convention gap is exactly sqrt(3) on the velocity at every (A, g, lam)
        assert (rc.boussinesq_velocity_alt(atwood, g, lam)
                / rc.boussinesq_velocity(atwood, g, lam)) == pytest.approx(math.sqrt(3.0), rel=1e-14)


def test_two_fluid_branches_split_by_atwood_only() -> None:
    """The split the Boussinesq model cannot produce is sqrt((1+A)/(1-A))."""
    a, g, lam = 1 / 3, 2e-4, 64
    ub = rc.goncharov_velocity(a, g, lam, "b")
    us = rc.goncharov_velocity(a, g, lam, "s")
    assert us / ub == pytest.approx(math.sqrt((1 + a) / (1 - a)), rel=1e-14)
    assert us / ub == pytest.approx(math.sqrt(2.0), rel=1e-3)     # 1.414 at A = 1/3


# ---------------------------------------------------------------------------
# 2. the bubble <-> spike symmetry of the single-density Boussinesq model
# ---------------------------------------------------------------------------

#: c_y -> -c_y on D2Q9: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1)
#: 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
FLIP_CY = [0, 1, 4, 3, 2, 8, 7, 6, 5]


def _mirror_state(f: torch.Tensor, phi: torch.Tensor):
    """Apply (phi, uy, y) -> (-phi, -uy, -y) to a discrete LBM state."""
    f_m = f[FLIP_CY].flip(1).contiguous()
    phi_m = (-phi.flip(0)).contiguous()
    w = W.to(f.device).view(9, 1, 1)
    return f_m, w * phi_m.unsqueeze(0), phi_m


def _initial_state(ny: int, nx: int, lam: int):
    """Deliberately asymmetric, bounded, smooth state (no eigenmode seeding)."""
    y = torch.arange(ny).view(-1, 1).float()
    x = torch.arange(nx).view(1, -1).float()
    phi = (0.8 * torch.tanh((y - 0.35 * ny) / 2.0)
           + 0.15 * torch.sin(2 * math.pi * x / nx + 0.3)
           + 0.12 * (y / ny) ** 2)          # breaks the mirror antisymmetry
    ux = 0.01 * torch.sin(2 * math.pi * x / nx) * torch.cos(math.pi * y / ny)
    uy = (0.01 * torch.cos(2 * math.pi * x / nx + 0.7)
          + 0.004 * torch.sin(math.pi * y / ny) + 0.0 * x)
    rho = 1.0 + 0.02 * torch.sin(math.pi * y / ny) + 0.0 * x
    f = equilibrium(rho, ux, uy)
    w = W.to(f.device).view(9, 1, 1)
    return f, w * phi.unsqueeze(0), phi


@pytest.mark.parametrize("scheme", ["bgk", "allen_cahn"])
def test_boussinesq_rt_is_bubble_spike_symmetric(scheme: str) -> None:
    """A mirrored run must stay the mirror of the original to round-off.

    If this holds, bubble and spike grow identically and no Atwood asymmetry
    (``u_s/u_b = sqrt((1+A)/(1-A))``) can ever be measured for this model -- the
    gate that round 1 called "lost asymmetry" (its finding F2) is not a defect of
    the interface scheme, it is the model.
    """
    torch.set_num_threads(2)
    torch.manual_seed(0)
    ny, nx, lam, n_steps = 16, 16, 16, 60
    cfg = RTConfig(lam=lam, layers=ny / lam, g=2e-4, atwood=1 / 3, tau=0.515,
                   tau_phi=0.501, n_steps=n_steps, sample_every=1,
                   phase_scheme=scheme, ac_sigma=3e-4, ac_width=2.0, ac_mobility=20.0)
    cfg.validate()
    f, g, phi = _initial_state(ny, nx, lam)
    fm, gm, phim = _mirror_state(f, phi)
    # the initial condition is genuinely asymmetric: mirroring it changes it
    # the state must NOT already be in the antisymmetric (invariant) subspace:
    # phi compared with its own mirror image -(-phi.flip) = phi.flip
    assert (phi + phi.flip(0)).abs().max().item() > 0.05, "initial state is mirror-symmetric"

    for _ in range(n_steps):
        f, g = rt_step(f, g, cfg, phi=phi)
        phi = g.sum(0)
        fm, gm = rt_step(fm, gm, cfg, phi=phim)
        phim = gm.sum(0)

    dphi = (phim + phi.flip(0)).abs().max().item()
    rho, ux, uy = macroscopic(f)
    rho_m, ux_m, uy_m = macroscopic(fm)
    duy = (uy_m + uy.flip(0)).abs().max().item()
    dphi0 = max(phi.abs().max().item(), 1e-12)
    assert dphi / dphi0 < 1e-4, f"{scheme}: mirror symmetry broken, dphi={dphi:.3e}"
    assert duy < 1e-4, f"{scheme}: mirror symmetry broken, duy={duy:.3e}"
    assert (rho_m - rho.flip(0)).abs().max().item() < 1e-4


def test_front_metrics_are_symmetric_under_the_mirror() -> None:
    """The measured front estimators must inherit the symmetry (u_s == u_b)."""
    phi = torch.zeros(32, 16)
    y = torch.arange(32).view(-1, 1).float()
    x = torch.arange(16).view(1, -1).float()
    phi = -torch.tanh((y - 16.0 - 3.0 * torch.cos(2 * math.pi * x / 16)) / 1.5)
    fm = rc.front_metrics(phi)
    fm2 = rc.front_metrics(-phi.flip(0))
    assert fm["h_bubble_col"] == pytest.approx(fm2["h_spike_col"], rel=1e-12, abs=1e-12)
    assert fm["h_spike_col"] == pytest.approx(fm2["h_bubble_col"], rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. the re-grade published with round 3 must be reproducible from raw runs
# ---------------------------------------------------------------------------


def test_round3_regrade_reproduces_from_history() -> None:
    """``artifacts/round3/regrade.json`` must be recomputable from ``history.csv``.

    Guards against hand-edited tables: every published error is recomputed here
    from the raw per-step records, and the two must agree to 1e-9 relative.
    """
    regrade_path = BENCH / "artifacts" / "round3" / "regrade.json"
    if not regrade_path.is_file():
        pytest.skip("round-3 artifacts not present in this checkout")
    try:
        import postprocess_round3 as pp
    except ModuleNotFoundError:                       # pragma: no cover
        pytest.skip("postprocess_round3.py not present")
    published = json.loads(regrade_path.read_text())
    rows = published["rows"]
    assert rows, "empty regrade table"
    for row in rows:
        case_dir = pp.find_case_dir(row["case"], row.get("source_root"))
        assert case_dir is not None, f"raw run missing for {row['case']}"
        fresh = pp.evaluate(case_dir)
        for key in ("u_front", "u_ref_primary", "err_primary_pct",
                    "u_ref_alt", "err_alt_pct", "sigma_sim"):
            got, want = fresh[key], row[key]
            if isinstance(want, float) and math.isnan(want):
                assert math.isnan(got)
            else:
                assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (row["case"], key)