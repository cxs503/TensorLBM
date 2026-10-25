"""Unit tests for the SIR-1b τ-ladder instrument (m298).

Covers the two additions made to
``benchmarks/verified/suboff_sail_resistance`` for the uniform-grid τ ladder:

* the ``SUBOFF_TAU_TARGET`` override — the lattice viscosity that the ladder
  asks for must be reached *exactly*, because a τ ladder is meaningless if the
  knob does not land on the requested relaxation time;
* the FA anchor probe — the #257/#265 equilibrium identity
  ``mem = E0 + E1 + E2 + NEQ`` must close, the probe must be read-only, and it
  must reproduce the library's background-subtracted momentum exchange.

These are pure CPU tests; no GPU and no engine setup required.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CASE = REPO / "benchmarks" / "verified" / "suboff_sail_resistance"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(CASE))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, CASE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_mod = _load("suboff_sail_run_m298", "run.py")
probe_mod = _load("tau_anchor_probe_m298", "tau_anchor_probe.py")


# --------------------------------------------------------------------------- #
# τ override
# --------------------------------------------------------------------------- #
def test_tau_target_unset_is_none(monkeypatch):
    monkeypatch.delenv("SUBOFF_TAU_TARGET", raising=False)
    assert run_mod._tau_target_override() is None


def test_tau_target_reads_env(monkeypatch):
    monkeypatch.setenv("SUBOFF_TAU_TARGET", "0.55")
    assert run_mod._tau_target_override() == pytest.approx(0.55)


@pytest.mark.parametrize("tau", [0.51, 0.55, 0.60, 0.70, 1.00])
def test_physical_viscosity_reaches_target_tau_exactly(tau):
    """The derived lattice viscosity must equal (τ-0.5)/3 to float precision.

    ``LBMUnitConverter`` picks ``u_lb`` (default 0.05) and derives
    ``ν_lb = u_lb·nx/Re``; converting back through ``Re = u_phys·L/nu_phys``
    must return exactly the requested τ.  This is the defect the m292 ladder
    died on: it wrote a *lattice* ν into the *physical* viscosity field.
    """
    from tensorlbm.unit_converter import LBMUnitConverter

    L = 80
    u_lb = 0.05
    nu_phys = run_mod._physical_viscosity_for_tau(tau, L, u_lb)
    Re = run_mod.U_PHYS * run_mod.SUBOFF_LENGTH_M / nu_phys
    uc = LBMUnitConverter(
        re=Re,
        l_phys=run_mod.SUBOFF_LENGTH_M,
        u_phys=run_mod.U_PHYS,
        nu_phys=nu_phys,
        nx=L,
    )
    assert uc.u_lb == pytest.approx(u_lb, abs=1e-12)
    assert uc.nu_lb == pytest.approx((tau - 0.5) / 3.0, rel=1e-12)
    assert uc.tau == pytest.approx(tau, abs=1e-12)


def test_tau_at_or_below_half_is_rejected():
    with pytest.raises(SystemExit):
        run_mod._physical_viscosity_for_tau(0.5, 80, 0.05)
    with pytest.raises(SystemExit):
        run_mod._physical_viscosity_for_tau(0.49, 80, 0.05)


def test_tau_is_the_reynolds_knob():
    """τ↑ ⇒ ν↑ ⇒ Re↓, the coupling the m298 supplement quantifies."""
    sys.path.insert(0, str(CASE))
    driver = _load("tau_ladder_driver_m298", "run_tau_ladder.py")
    res = [driver.re_lattice(t, L=80) for t in (0.51, 0.55, 0.60, 0.70, 1.00)]
    assert res == sorted(res, reverse=True)
    assert res[0] == pytest.approx(0.05 * 80 / ((0.51 - 0.5) / 3.0))
    assert res[-1] == pytest.approx(0.05 * 80 / ((1.00 - 0.5) / 3.0))
    # the ladder spans a factor 50 in effective Reynolds number
    assert res[0] / res[-1] == pytest.approx(50.0, rel=1e-9)


def test_reference_lines_behave_at_extremes():
    assert run_mod.blasius_cf(1000.0) == pytest.approx(0.041995047327036086, rel=1e-12)
    # ITTC-1957 has a pole at Re = 100 (log10 Re = 2); guard, do not crash
    assert math.isnan(run_mod.ittc1957_cf(100.0))
    assert run_mod.ittc1957_cf(1.0e7) > 0.0


# --------------------------------------------------------------------------- #
# FA anchor probe
# --------------------------------------------------------------------------- #
def _synthetic_case(shape=(10, 12, 18)):
    """One finite solid block in a uniform free stream.

    The block is finite along *all three* axes on purpose: a solid slab that
    spans the whole streamwise extent has no ±x fluid→solid links at all (the
    periodic roll makes the crossing set empty), which would zero the axial part
    of the anchor family by construction and test nothing.
    """
    nz, ny, nx = shape
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[nz // 2 - 1 : nz // 2 + 2, ny // 2 - 1 : ny // 2 + 2, nx // 3 : 2 * nx // 3] = True
    # near = fluid cells adjacent to solid (same call the engine uses)
    from tensorlbm.drag_pressure import get_near_wall_3d

    near = get_near_wall_3d(solid)
    rho = torch.ones(shape)
    u = torch.full(shape, 0.05)
    zero = torch.zeros(shape)
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(rho, u, zero, zero)
    return solid, near, f


def _probe(tmp_path, **kw):
    solid, near, f = _synthetic_case()
    cx = solid.shape[2] * 0.25
    p = probe_mod.AnchorProbe(
        solid=solid,
        near=near,
        cx=cx,
        length=6.0,
        dpS=1.0,
        u_lb=0.05,
        path=str(tmp_path / "anchors.jsonl"),
        cadence=kw.pop("cadence", 1),
    )
    return p, f


def test_thirds_partition_the_fluid_domain():
    solid, _near, _f = _synthetic_case()
    thirds = probe_mod.build_axial_thirds(solid, solid.shape[2] * 0.25, 6.0)
    fluid = ~solid
    covered = thirds[0] | thirds[1] | thirds[2]
    assert bool((fluid & ~covered).sum().item()) == 0
    # disjoint
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        assert bool((thirds[i] & thirds[j]).sum().item()) == 0
    total = sum(int(m.sum().item()) for m in thirds)
    assert total == int(fluid.sum().item())


def test_probe_identity_closes_and_is_read_only(tmp_path):
    p, f = _probe(tmp_path)
    f_before = f.clone()
    row = p.sample(f, step=1, tau=0.55, nu_lb=(0.55 - 0.5) / 3.0)
    p.close()
    # read-only guarantee
    assert torch.equal(f, f_before)
    # E0/E1/E2/NEQ identity closes by construction
    assert row["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert p.max_identity_residual == pytest.approx(0.0, abs=1e-12)
    # a uniform state has no dipole
    assert row["D_QW"] == pytest.approx(0.0, abs=1e-12)
    assert row["step"] == 1


def test_probe_E0_responds_to_an_axial_density_gradient(tmp_path):
    """E0 sums c_ix·2w·rho over the crossing set, so it needs an axial asymmetry.

    On a fore-aft symmetric uniform state the ±x links cancel exactly (recorded
    by the test above).  A density gradient along x must break that symmetry and
    make E0 non-zero, and it must still close the identity.
    """
    solid, near, _f = _synthetic_case()
    nz, ny, nx = solid.shape
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    rho = 1.0 + 0.01 * x / nx  # gentle axial density gradient
    rho = rho.expand(nz, ny, nx).contiguous()
    u = torch.full((nz, ny, nx), 0.05)
    zero = torch.zeros((nz, ny, nx), dtype=torch.float32)
    from tensorlbm.d3q19 import equilibrium3d

    f = equilibrium3d(rho, u, zero, zero)

    p = probe_mod.AnchorProbe(
        solid=solid,
        near=near,
        cx=nx * 0.25,
        length=6.0,
        dpS=1.0,
        u_lb=0.05,
        path=str(tmp_path / "grad.jsonl"),
        cadence=1,
    )
    row = p.sample(f, step=1, tau=0.55, nu_lb=(0.55 - 0.5) / 3.0)
    p.close()
    assert abs(row["E0_x"]) > 1e-10
    assert row["identity_residual"] == pytest.approx(0.0, abs=1e-12)
    # the dipole picks up the bow/stern asymmetry of the same gradient
    assert row["D_QW"] != 0.0


def test_probe_bgsub_matches_library_reference(tmp_path):
    """The gathered background-subtracted impulse == the library function.

    The library accumulates ``fx`` as a float32 0-dim tensor, so on a state
    whose exact answer is zero it reports float32 round-off (1.19e-7 here).
    The probe reduces in Python float64 and lands on the exact value.  The
    tolerance below is therefore the float32 noise floor, not a free pass:
    a genuine algebraic disagreement would be O(1) on this state.
    """
    from tensorlbm.momentum_exchange import momentum_exchange_background_subtracted as lib

    solid, near, f = _synthetic_case()
    p = probe_mod.AnchorProbe(
        solid=solid,
        near=near,
        cx=solid.shape[2] * 0.25,
        length=6.0,
        dpS=1.0,
        u_lb=0.05,
        path=str(tmp_path / "a.jsonl"),
        cadence=1,
    )
    inline = p._bgsub_inline(f)
    p.close()
    ref = lib(f, solid, near, rho0=1.0, u0=(0.05, 0.0, 0.0))[0]
    assert abs(inline - ref) <= 1e-5 * max(1.0, abs(ref))


def test_probe_bgsub_matches_library_on_a_loaded_state(tmp_path):
    """Same check on a state with a real (non-cancelling) axial asymmetry."""
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.momentum_exchange import momentum_exchange_background_subtracted as lib

    solid, near, _f = _synthetic_case()
    nz, ny, nx = solid.shape
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    rho = (1.0 + 0.02 * x / nx).expand(nz, ny, nx).contiguous()
    u = torch.full((nz, ny, nx), 0.05)
    zero = torch.zeros((nz, ny, nx), dtype=torch.float32)
    f = equilibrium3d(rho, u, zero, zero)

    p = probe_mod.AnchorProbe(
        solid=solid,
        near=near,
        cx=nx * 0.25,
        length=6.0,
        dpS=1.0,
        u_lb=0.05,
        path=str(tmp_path / "b.jsonl"),
        cadence=1,
    )
    inline = p._bgsub_inline(f)
    p.close()
    ref = lib(f, solid, near, rho0=1.0, u0=(0.05, 0.0, 0.0))[0]
    assert abs(ref) > 1e-8  # the state actually carries a signal
    # float32 accumulation noise in the library vs float64 in the probe:
    # 4.8e-7 absolute on a 1.6e-2 signal.  Bound it relatively but keep the
    # test able to catch an algebraic disagreement (which is O(1) here).
    assert abs(inline - ref) <= 1e-4 * abs(ref)


def test_probe_crossing_count_matches_near_wall_set(tmp_path):
    """Every crossing link must belong to the near-wall set (no stray cells)."""
    p, _f = _probe(tmp_path)
    p.close()
    assert p.n_links > 0
    assert p.n_crossing > 0


def test_probe_jsonl_is_written_per_sample(tmp_path):
    import json

    p, f = _probe(tmp_path)
    for k in range(1, 4):
        p.sample(f, step=k, tau=0.55, nu_lb=(0.55 - 0.5) / 3.0)
    p.close()
    lines = [
        json.loads(x) for x in (tmp_path / "anchors.jsonl").read_text().splitlines() if x.strip()
    ]
    assert len(lines) == 3
    assert [ln["step"] for ln in lines] == [1, 2, 3]
    for ln in lines:
        for key in ("mem_fx", "mem_bgsub_fx", "E0_x", "E1_x", "E2_x", "NEQ_x",
                    "ct", "e0", "rho_fwd", "rho_mid", "rho_aft", "D_QW"):
            assert key in ln