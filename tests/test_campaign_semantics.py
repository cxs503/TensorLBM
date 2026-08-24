"""B3 stage 6: campaign-semantics alignment knobs (semantic regression tests).

Each test pins one definitional knob of :class:`CampaignSemantics` to the
production campaign chain it names, at the strongest available level —
bitwise equality against the production operator, or the exact step-number
bookkeeping of the mass correction (the phase/interval semantics of
``scan_runner.run_scan_point``, the same per-step impulse attribution the
drag survey compensates — cf. the ``note_mass_correction`` tests in
``tests/test_scan_drag.py``).
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.autograd_calib import (
    CAMPAIGN_SEMANTICS,
    CUMULANT27_RATES,
    DEFAULT_MRT_RATES,
    LEGACY27_SEMANTICS,
    LEGACY_SEMANTICS,
    CampaignSemantics,
    HullCase,
    _initial_f19,
    _initial_f27,
    _plane_shell,
    _rate27_collide,
    _rate_collide,
    campaign_chain19,
    campaign_chain27,
    campaign_rollout19,
    press_profile,
    press_profile_campaign,
    step27,
)
from tensorlbm.autograd_path import InletSpec, OutletSpec, WallSpec, differentiable_step
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d

_NZ, _NY, _NX = 8, 10, 24


def _box(**kw) -> HullCase:
    base = dict(steps=40, window_start=20, device="cpu", dtype=torch.float32)
    base.update(kw)
    return HullCase(_NZ, _NY, _NX, **base)


def _small_mask(box: HullCase) -> torch.Tensor:
    mask = torch.zeros((box.nz, box.ny, box.nx), dtype=torch.bool)
    mask[:, box.ny // 2 - 1 : box.ny // 2 + 1, 2 * box.nx // 8 : 5 * box.nx // 8] = True
    return mask


def _cumulant(box: HullCase, re: float):
    tau = box.tau_of_re(re)
    return tau, (lambda f, _t: collide_cumulant_d3q19(f, tau))


# -- knob 1: initialisation inside the solid --------------------------------


def test_init_rest_matches_campaign_initial_f_bitwise() -> None:
    """``init_solid='rest'`` reproduces ``suboff_n128.initial_f`` bit-for-bit."""
    from tensorlbm.cases import get_case

    box = _box()
    case = get_case("suboff_n128", resolution=(box.nz, box.ny, box.nx), re=100.0, device="cpu")
    assert case.u_in == box.u_in
    mask = case.solid_mask()  # the production hull mask (== box.make_mask(), stage 4)
    f_sem = _initial_f19(box, mask, CampaignSemantics(init_solid="rest"))
    f_case = case.initial_f()
    assert f_sem.shape == f_case.shape
    assert torch.equal(f_sem, f_case), "rest init must equal the campaign initial_f bitwise"


def test_init_freestream_is_solid_agnostic() -> None:
    """Legacy init ignores the mask (free-stream equilibrium everywhere)."""
    box = _box()
    mask = _small_mask(box)
    f0 = _initial_f19(box, mask, LEGACY_SEMANTICS)
    f_no_mask = _initial_f19(box, torch.zeros_like(mask), LEGACY_SEMANTICS)
    assert torch.equal(f0, f_no_mask)
    # equilibrium populations sum to rho = 1 per cell, solid included
    assert torch.allclose(f0.sum(dim=0), torch.ones_like(f0.sum(dim=0)))


def test_initial_f27_rest_zero_inside_solid() -> None:
    box = _box()
    mask = _small_mask(box)
    f = _initial_f27(box, mask, CampaignSemantics(init_solid="rest"))
    from tensorlbm.d3q27 import macroscopic27

    _, ux, _, _ = macroscopic27(f)
    assert torch.all(ux[mask] == 0.0)
    assert torch.allclose(ux[~mask], torch.full_like(ux[~mask], box.u_in))


# -- knobs 2/3: chain definitions -------------------------------------------


def test_legacy_chain19_is_differentiable_step_bitwise() -> None:
    """All-legacy knobs reproduce ``differentiable_step`` bit-for-bit."""
    box = _box()
    mask = _small_mask(box)
    tau = box.tau_of_re(120.0)
    collide = _rate_collide(tau, DEFAULT_MRT_RATES)
    step = campaign_chain19(box, collide, mask, LEGACY_SEMANTICS, tau)
    torch.manual_seed(0)
    f = torch.rand((19, box.nz, box.ny, box.nx), dtype=torch.float32) * 0.02 + 0.04
    inlet = InletSpec(ux=torch.as_tensor(box.u_in), method=box.inlet_method)
    walls = WallSpec(method=box.wall_method, ux=box.u_in)
    f_new, probe = step(f)
    ref_new, ref_probe = differentiable_step(
        f,
        tau,
        mask,
        collide=collide,
        inlet=inlet,
        outlet=OutletSpec(),
        walls=walls,
        return_probe=True,
    )
    assert torch.equal(f_new, ref_new)
    assert torch.equal(probe, ref_probe)


def test_campaign_chain19_bc_is_far_field_bc_3d_bitwise() -> None:
    """``outlet='full'`` + bounce reproduces the production BC bit-for-bit."""
    box = _box()
    mask = _small_mask(box)
    tau = box.tau_of_re(120.0)
    collide = _rate_collide(tau, DEFAULT_MRT_RATES)
    sem = CampaignSemantics(outlet="full", collide_solid=True)
    step = campaign_chain19(box, collide, mask, sem, tau)
    torch.manual_seed(1)
    f = torch.rand((19, box.nz, box.ny, box.nx), dtype=torch.float32) * 0.02 + 0.04
    f_new, probe = step(f)
    # campaign chain: collide everywhere -> stream -> far-field faces -> bounce
    f_bc = far_field_bc_3d(
        stream3d(collide(f, tau)),
        box.u_in,
        obstacle_mask=None,
        bc_config={"far_field_faces": ["y-", "y+", "z-", "z+"], "periodic_faces": []},
    )
    assert torch.equal(probe, f_bc)
    assert torch.equal(f_new, bounce_back_cells_3d(f_bc, mask))


def test_campaign_chain27_legacy_is_step27_bitwise() -> None:
    """All-legacy 27 knobs reproduce ``step27`` bit-for-bit."""
    from tensorlbm.d3q27 import equilibrium27

    box = _box()
    mask = _small_mask(box)
    tau = box.tau_of_re(120.0)
    collide = _rate27_collide(tau, CUMULANT27_RATES, "cumulant")
    step = campaign_chain27(box, collide, mask, LEGACY27_SEMANTICS, tau)
    torch.manual_seed(2)
    rho = torch.ones((box.nz, box.ny, box.nx))
    u0 = torch.zeros((3, box.nz, box.ny, box.nx))
    u0[0] = box.u_in
    f = equilibrium27(rho, u0[0], u0[1], u0[2]) + 0.01 * torch.rand((27, box.nz, box.ny, box.nx))
    f_new, probe = step(f)
    ref_new, ref_probe = step27(f, tau, mask, collide, box.u_in, return_probe=True)
    assert torch.equal(f_new, ref_new)
    assert torch.equal(probe, ref_probe)


def test_chain27_rejects_unknown_outlet() -> None:
    box = _box()
    mask = _small_mask(box)
    collide = _rate_collide(box.tau_of_re(100.0), DEFAULT_MRT_RATES)
    with pytest.raises(ValueError, match="full-copy outlet"):
        campaign_chain27(box, collide, mask, CampaignSemantics(outlet="unknown"))


# -- knob 4: mass-correction phase / interval / impulse bookkeeping ---------


def test_correction_step_numbers_follow_campaign_phase() -> None:
    """Corrections land exactly on the campaign step grid (scan_runner.py:874).

    Campaign: 1-indexed loop, ``step % 10 == 0`` from step 10 (transient
    included).  The stage-5 diagnosis configuration (every 20, window only)
    is the same predicate with shifted first/phase.
    """
    sem = CampaignSemantics(mass_every=10, mass_phase=0, mass_first=10)
    assert [s for s in range(1, 61) if sem.correct_at(s)] == [10, 20, 30, 40, 50, 60]
    # stage-5 diagnosis emulation: every 20, first correction at window_start+20
    sem_diag = CampaignSemantics(mass_every=20, mass_phase=0, mass_first=3820)
    assert [s for s in range(1, 4001) if sem_diag.correct_at(s)] == list(range(3820, 4001, 20))
    assert LEGACY_SEMANTICS.correct_at(10) is False
    with pytest.raises(ValueError):
        CampaignSemantics(mass_every=10, mass_phase=10)


def test_correction_restores_mass_and_dispatch_reads_post_correction() -> None:
    """End-to-end bookkeeping: the correction restores m0 and the frame is post-correction.

    The campaign dispatch phase (scan_runner.py:874-892) corrects *then*
    exports; the recorded frame profile must equal the post-rescale state,
    not the pre-correction probe — the same per-step impulse attribution the
    drag survey compensates (``DragSurveyObserver.note_mass_correction``).
    """
    box = _box(steps=10, window_start=0)
    mask = _small_mask(box)
    tau, collide = _cumulant(box, 150.0)
    sem = CampaignSemantics(
        init_solid="rest",
        collide_solid=True,
        outlet="full",
        mass_every=10,
        observe_frames=(10,),
    )
    out = campaign_rollout19(box, collide, sem, mask=mask, bins=8, chunk=10)
    # m0 is the initial total mass (equilibrium sums to rho = 1 per cell)
    assert out.m0 == pytest.approx(box.nz * box.ny * box.nx, rel=1e-5)
    # replay the chain manually to the correction step, then export
    step = campaign_chain19(box, collide, mask, sem, tau)
    with torch.no_grad():
        f = _initial_f19(box, mask, sem)
        for _ in range(10):
            f, _probe = step(f)
        f = correct_mass3d(f, out.m0)
        rho_ref = macroscopic3d(f)[0]
    cz, ays, axs, bin_idx = _plane_shell(mask, 8)
    counts = torch.bincount(bin_idx, minlength=8).clamp(min=1).double()
    ref = (
        torch.zeros(8, dtype=torch.float64).index_add(
            0, bin_idx, (rho_ref[cz][ays, axs] - 1.0).double()
        )
        / counts
    )
    assert torch.allclose(out.frames[10], ref, atol=1e-9), (
        "frame must be the post-correction export state"
    )
    assert out.n_window == 10


# -- knob 5: observation readout + full-driver legacy continuity -------------


def test_legacy_driver_matches_press_profile_bitwise() -> None:
    """The driver at all-legacy knobs is ``press_profile`` bit-for-bit."""
    box = _box(steps=40, window_start=20)
    mask = _small_mask(box)
    p_ref, cd_ref = press_profile(box, 150.0, DEFAULT_MRT_RATES, mask=mask, bins=8, chunk=20)
    collide = _rate_collide(box.tau_of_re(150.0), DEFAULT_MRT_RATES)
    out = campaign_rollout19(box, collide, LEGACY_SEMANTICS, mask=mask, bins=8, chunk=20)
    assert torch.equal(out.window_profile, p_ref)
    assert out.window_cd == pytest.approx(cd_ref, rel=1e-12)
    assert out.frames == {}


def test_frames_readout_selects_snapshot_definition() -> None:
    """observe_frames switches the readout to the campaign snapshot profile."""
    box = _box(steps=20, window_start=0)
    mask = _small_mask(box)
    collide = _rate_collide(box.tau_of_re(150.0), DEFAULT_MRT_RATES)
    sem = LEGACY_SEMANTICS.with_frames(20)
    out = campaign_rollout19(box, collide, sem, mask=mask, bins=8, chunk=20)
    assert set(out.frames) == {20}
    assert torch.linalg.vector_norm(out.frames[20]) > 0.0
    p, cd = press_profile_campaign(box, 150.0, DEFAULT_MRT_RATES, sem, mask=mask, bins=8)
    assert torch.equal(p, out.frames[20])
    assert cd == pytest.approx(out.window_cd, rel=1e-12)


def test_frame_beyond_steps_rejected() -> None:
    box = _box(steps=10, window_start=0)
    mask = _small_mask(box)
    sem = LEGACY_SEMANTICS.with_frames(11)
    with pytest.raises(ValueError, match="beyond box.steps"):
        campaign_rollout19(
            box, _rate_collide(box.tau_of_re(100.0), DEFAULT_MRT_RATES), sem, mask=mask, bins=8
        )


def test_campaign_semantics_constant_is_the_full_alignment() -> None:
    """The shipped constant names every knob at its campaign value."""
    assert CAMPAIGN_SEMANTICS.init_solid == "rest"
    assert CAMPAIGN_SEMANTICS.collide_solid is True
    assert CAMPAIGN_SEMANTICS.outlet == "full"
    assert (
        CAMPAIGN_SEMANTICS.mass_every,
        CAMPAIGN_SEMANTICS.mass_phase,
        CAMPAIGN_SEMANTICS.mass_first,
    ) == (10, 0, 10)
