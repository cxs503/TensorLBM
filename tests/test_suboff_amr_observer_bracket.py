"""Observer-bracket regression: the finest-CV hook blindness and its fix.

Proven defect (``diag_negative_force_20261011``, questions Q2/Q3): the
runner's finest-CV observer brackets only the L1 substep advances, so a
momentum impulse applied inside the pre-restrict hook window (leaf BFL +
restriction write-back + reflux) lands BETWEEN the before/out
measurement pairs and is structurally invisible — the reading collapses
to an annulus/seal bookkeeping residual whose sign is not set by the
wall; with ``fine_wall_chains=False`` the friction/pressure reporters
are additionally wired to the absent block wall chain and store literal
0.0 (Q3).

Locked directions on a synthetic impulse (the m4 unit-test style):

1. observer math: a fake substep sequence whose only momentum edit is a
   known injection inside the hook window — the legacy per-substep
   formula reads exactly 0.0 (blind), the full-root-step formula recovers
   the injected impulse to machine precision (library-identity equality,
   no tolerance), removal reading +drag and source reading -drag;
2. runner integration (small CPU composition, hook injecting a known
   population edit inside the finest CV): legacy artifacts keep
   ``Ct_fric == Ct_wall_total == 0.0`` exactly; ``full_step`` mode
   captures the injection (finest-CV identity against the independently
   reduced hook impulse), populates the leaf-ledger keys from the mailbox
   exactly, and leaves every legacy series bit-for-bit unchanged
   (measurement only, no physics change);
3. v1/v2 record reproduction: a legacy-mode hooked run is bitwise
   identical to the frozen tree (1c759d2) once the additive F-6 keys are
   stripped — skipped when the frozen tree is absent.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm.control_volume_force import (
    box_control_volume,
    fluid_momentum_change,
    observe_control_volume_force,
)
from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.suboff_amr_wallmodel_runner import (
    LEAF_AREA_FACTOR,
    LeafForceLedgerMailbox,
    SuboffAMRWallModelConfig,
    _cv_crop_window,
    build_amr_contexts,
    run_suboff_amr_wallmodel,
)

FROZEN_TREE = Path("/nfs/wangxi/wt_highre")

# posted leaf ledger values (leaf LU, per hook firing; link = body/+drag,
# shear = fluid frame -> body-frame friction = -shear)
LEAF_LINK_POST = 0.3
LEAF_SHEAR_POST = -0.02

# additive keys (F-6 observer, v3.3 leaf post-wall positivity):
# present on this branch, absent from the frozen tree
_ADDITIVE_RESULT_KEYS = (
    "observer_mode",
    "ct_cv_full_step",
    "ct_root_envelope_cv_full_step",
    "ct_fric_leaf",
    "ct_wall_total_leaf",
    "ct_leaf_total_cadence_p",
    "ct_series_observers",
    "leaf_ledger_convention",
    "full_step_observer",
    "leaf_post_wall_positivity",
)
_ADDITIVE_SERIES_KEYS = (
    "observer_mode",
    "ct_cv_full_step",
    "ct_root_envelope_cv_full_step",
    "ct_fric_leaf",
    "ct_wall_total_leaf",
    "ct_leaf_total_cadence_p",
)


def _strip_additive_keys(artifact: dict) -> dict:
    stripped = copy.deepcopy(artifact)
    for key in _ADDITIVE_RESULT_KEYS:
        stripped["result"].pop(key, None)
    stripped["configuration"].pop("observer_mode", None)
    stripped["configuration"].pop("leaf_post_wall_positivity", None)
    for entry in stripped["ct_time_series"]:
        for key in _ADDITIVE_SERIES_KEYS:
            entry.pop(key, None)
    return stripped


# ---------------------------------------------------------------------------
# 1. observer math (m4 style): legacy blind, full-root-step exact
# ---------------------------------------------------------------------------
N = 14
U = 0.05
DELTA = 2.0e-3
CV_BOUNDS = dict(x0=4, x1=10, y0=4, y1=10, z0=4, z1=10)
N_SUBSTEPS = 2  # one root step = two identity fine substeps + the hook edit
_Q_POS_X = int(torch.nonzero((C[:, 0] == 1) & (C[:, 1] == 0) & (C[:, 2] == 0)).flatten()[0])


def _uniform_equilibrium() -> torch.Tensor:
    rho = torch.ones((N, N, N))
    ux = torch.full_like(rho, U)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, ux, zero, zero, device="cpu")


def _cv() -> torch.Tensor:
    return box_control_volume((N, N, N), **CV_BOUNDS, device="cpu")


def _fluid_mx(f: torch.Tensor, cv: torch.Tensor) -> float:
    """Independent float64 inventory reduction (the m4 cross-check)."""
    inventory = (f.to(torch.float64) * cv.unsqueeze(0)).sum(dim=(1, 2, 3))
    return float((inventory.unsqueeze(1) * C.to(torch.float64)).sum().item())


def _hook_edited_field(sign: float) -> torch.Tensor:
    """Exit state: the hook window's edit is the ONLY momentum change."""
    f = _uniform_equilibrium()
    f[_Q_POS_X, 7, 7, 7] += sign * DELTA
    return f


@pytest.mark.parametrize("sign", [-1.0, +1.0])
def test_legacy_formula_is_blind_to_the_hook_impulse(sign: float) -> None:
    f_entry = _uniform_equilibrium()
    f_exit = _hook_edited_field(sign)
    cv = _cv()

    # the fake substep sequence: identity advances (before == post == out),
    # so every legacy per-substep pair sees zero change and zero import
    substep_forces = [
        float(observe_control_volume_force(f_entry, f_entry, f_entry, cv).force_on_body[0].item())
        for _ in range(N_SUBSTEPS)
    ]
    assert substep_forces == [0.0, 0.0]  # exactly blind: ~0 residual
    legacy_reading = sum(substep_forces) / len(substep_forces)
    assert legacy_reading == 0.0

    # the hook edit is real fluid momentum: removed (sign -1) / added (+1)
    edited = _fluid_mx(f_exit, cv) - _fluid_mx(f_entry, cv)
    assert math.copysign(1.0, edited) == sign
    assert edited != 0.0


@pytest.mark.parametrize("sign,expected_force_sign", [(-1.0, +1.0), (+1.0, -1.0)])
def test_full_step_formula_captures_the_hook_impulse_exactly(
    sign: float, expected_force_sign: float
) -> None:
    f_entry = _uniform_equilibrium()
    f_exit = _hook_edited_field(sign)
    cv = _cv()

    imports_sum = sum(
        float(observe_control_volume_force(f_entry, f_entry, f_entry, cv).momentum_import[0].item())
        for _ in range(N_SUBSTEPS)
    )
    change = fluid_momentum_change(f_entry, f_exit, cv, periodic_axes=())
    total_capture = imports_sum - float(change[0])
    per_root_force = total_capture / N_SUBSTEPS

    edited = _fluid_mx(f_exit, cv) - _fluid_mx(f_entry, cv)
    # machine precision, identity not tolerance: the full-root-step bracket
    # recovers the injected impulse exactly (library reductions of the same
    # edit), with the drag-positive sign convention (removal -> +drag)
    assert total_capture == -edited
    assert per_root_force == -edited / N_SUBSTEPS
    assert math.copysign(1.0, per_root_force) == expected_force_sign


# ---------------------------------------------------------------------------
# 2. runner integration: production loop on a small CPU composition
# ---------------------------------------------------------------------------
def _integration_config(**overrides) -> SuboffAMRWallModelConfig:
    base: dict = dict(
        nx=96,
        ny=48,
        nz=48,
        hull_length=48.0,
        hull_type="BARE_HULL",
        l1_wall_margin=3,
        l1_wake_cells=8,
        cv_margin=2,
        steps=3,
        warmup_steps=0,
        ramp_steps=4,
        wall_diagnostic_interval=1,
        device="cpu",
        fine_wall_chains=False,
        seal_solid_mode="eroded_core",
        seal_erode_passes=3,
        root_bounce_seal=False,
    )
    base.update(overrides)
    return SuboffAMRWallModelConfig(**base)


def _make_injecting_hook(config: SuboffAMRWallModelConfig, delta: float, mailbox=None):
    """Hook injecting a known population edit inside the finest CV.

    The edit cell is picked deterministically from the geometry build
    (identical on every tree), and the expected capture — the negative of
    the hook's fluid-momentum change over the CV, the body-frame impulse —
    is recorded per firing via the library's own reduction.
    """
    built = build_amr_contexts(config, torch.device("cpu"))
    ghost = built["ghost"]
    context = built["contexts"][-1]
    cv = context.control_volume
    assert cv is not None
    fluid_in_cv = cv & ~context.solid & context.physical_mask
    cells = torch.nonzero(fluid_in_cv, as_tuple=False)
    assert len(cells) > 0
    zc, yc, xc = (int(v) for v in cells[len(cells) // 2])
    window = _cv_crop_window(cv)
    cv_crop = cv[window].clone()
    solid_crop = context.solid[window].clone()

    def _interior_crop(view: torch.Tensor) -> torch.Tensor:
        return view[
            :,
            window[0].start - ghost : window[0].stop - ghost,
            window[1].start - ghost : window[1].stop - ghost,
            window[2].start - ghost : window[2].stop - ghost,
        ]

    expected_captures: list[float] = []

    def hook(fine_old, fine_new, fine_post) -> None:
        pre = _interior_crop(fine_new).clone()
        fine_new[_Q_POS_X, zc - ghost, yc - ghost, xc - ghost] -= delta
        post = _interior_crop(fine_new).clone()
        expected_captures.append(
            -float(fluid_momentum_change(pre, post, cv_crop, solid=solid_crop, periodic_axes=())[0])
        )
        if mailbox is not None:
            mailbox.post(LEAF_LINK_POST, LEAF_SHEAR_POST)

    return hook, expected_captures


def test_legacy_mode_stores_exact_zeros_and_no_new_keys() -> None:
    config = _integration_config()
    hook, _ = _make_injecting_hook(config, DELTA)
    artifact = run_suboff_amr_wallmodel(config, pre_restrict_hook=hook)

    # Q3 locked: the dead has_wall_chain branch stores literal 0.0
    assert artifact["result"]["Ct_fric"] == 0.0
    assert artifact["result"]["Ct_wall_total"] == 0.0
    assert artifact["result"]["observer_mode"] == "legacy"
    assert math.isnan(artifact["result"]["ct_cv_full_step"])
    assert math.isnan(artifact["result"]["ct_fric_leaf"])
    assert artifact["result"]["ct_series_observers"]["ct_cv"] == "legacy_substep_bracket"
    for entry in artifact["ct_time_series"]:
        for key in _ADDITIVE_SERIES_KEYS:
            assert key not in entry
        assert entry["ct_fric"] == 0.0
        assert entry["ct_wall_total"] == 0.0


def test_full_step_mode_captures_injection_and_populates_leaf_ledger() -> None:
    config = _integration_config(observer_mode="full_step")
    mailbox = LeafForceLedgerMailbox()
    hook, expected_captures = _make_injecting_hook(config, DELTA, mailbox=mailbox)
    artifact = run_suboff_amr_wallmodel(config, pre_restrict_hook=hook, leaf_force_ledger=mailbox)

    assert artifact["result"]["observer_mode"] == "full_step"
    assert len(expected_captures) == config.steps  # one hook firing per root step

    # --- capture identity: (full - legacy) * denominator == hook impulse
    # per substep (the finest-CV bracket sees the between-pairs edit) ---
    denominator = 0.5 * config.lattice_speed**2 * artifact["result"]["reference_area"]
    n_substeps = 2  # levels=1 -> ratio 2 finest substeps per root step
    for entry, expected in zip(artifact["ct_time_series"], expected_captures, strict=True):
        capture = (entry["ct_cv_full_step"] - entry["ct_cv"]) * denominator
        # measured residual on this stack: <= 2.2e-14 relative (fp64
        # reduction-order only); 1e-10 leaves four orders of margin
        assert capture == pytest.approx(expected / n_substeps, rel=1e-10, abs=1e-14)
        assert math.isfinite(entry["ct_cv_full_step"])
        assert math.isfinite(entry["ct_root_envelope_cv_full_step"])

    # --- leaf ledger keys populate exactly (mailbox -> Ct mapping) ---
    denominator_leaf = denominator * LEAF_AREA_FACTOR
    n_entries = len(artifact["ct_time_series"])
    single = (-LEAF_SHEAR_POST) / denominator_leaf
    # replicate the runner's window-mean aggregation for exact comparison
    expected_ct_fric_leaf = sum([single] * n_entries) / n_entries
    single_wall = (LEAF_LINK_POST - LEAF_SHEAR_POST) / denominator_leaf
    expected_ct_wall_leaf = sum([single_wall] * n_entries) / n_entries
    single_total = (LEAF_LINK_POST + LEAF_SHEAR_POST) / denominator_leaf
    expected_ct_total = sum([single_total] * n_entries) / n_entries
    assert artifact["result"]["ct_fric_leaf"] == expected_ct_fric_leaf
    assert artifact["result"]["ct_fric_leaf"] != 0.0
    assert artifact["result"]["ct_wall_total_leaf"] == expected_ct_wall_leaf
    assert artifact["result"]["ct_wall_total_leaf"] != 0.0
    assert artifact["result"]["ct_leaf_total_cadence_p"] == expected_ct_total
    assert artifact["result"]["full_step_observer"]["full_step_samples"] == config.steps
    assert artifact["result"]["full_step_observer"]["leaf_ledger_samples"] == config.steps
    assert artifact["result"]["leaf_ledger_convention"]["link_force"]
    assert artifact["result"]["ct_series_observers"]["ct_fric_leaf"] == (
        "leaf_force_ledger_cadence_p"
    )
    for entry in artifact["ct_time_series"]:
        assert entry["observer_mode"] == "full_step"
        assert entry["ct_fric_leaf"] == single

    # --- Q3 on the legacy keys inside the new mode: still exactly 0.0 ---
    assert artifact["result"]["Ct_fric"] == 0.0
    assert artifact["result"]["Ct_wall_total"] == 0.0


def test_full_step_mode_changes_no_physics_legacy_series_bitwise() -> None:
    """Measurement only: the legacy series is bit-identical in both modes."""
    legacy_config = _integration_config()
    legacy_hook, _ = _make_injecting_hook(legacy_config, DELTA)
    legacy = run_suboff_amr_wallmodel(legacy_config, pre_restrict_hook=legacy_hook)

    full_config = _integration_config(observer_mode="full_step")
    full_hook, _ = _make_injecting_hook(full_config, DELTA)
    full = run_suboff_amr_wallmodel(full_config, pre_restrict_hook=full_hook)

    stripped_series = [
        {key: value for key, value in entry.items() if key not in _ADDITIVE_SERIES_KEYS}
        for entry in full["ct_time_series"]
    ]
    assert stripped_series == legacy["ct_time_series"]
    assert full["mx_finest_substeps"] == legacy["mx_finest_substeps"]
    assert full["mx_per_level_per_step"] == legacy["mx_per_level_per_step"]
    assert full["result"]["Ct_cv"] == legacy["result"]["Ct_cv"]
    assert full["result"]["Ct_root_envelope_cv"] == legacy["result"]["Ct_root_envelope_cv"]
    assert full["result"]["Ct_fric"] == legacy["result"]["Ct_fric"]
    assert full["result"]["Ct_wall_total"] == legacy["result"]["Ct_wall_total"]


def test_observer_mode_validation_and_empty_mailbox_read_nan() -> None:
    with pytest.raises(ValueError, match="observer_mode"):
        _integration_config(observer_mode="nonsense").validate()
    # full_step without a mailbox: leaf keys are NaN (unmeasured), never 0.0
    config = _integration_config(observer_mode="full_step")
    hook, _ = _make_injecting_hook(config, DELTA)
    artifact = run_suboff_amr_wallmodel(config, pre_restrict_hook=hook)
    assert math.isnan(artifact["result"]["ct_fric_leaf"])
    assert math.isnan(artifact["result"]["ct_wall_total_leaf"])
    assert math.isnan(artifact["result"]["ct_leaf_total_cadence_p"])
    assert artifact["result"]["full_step_observer"]["leaf_ledger_samples"] == 0
    assert math.isfinite(artifact["result"]["ct_cv_full_step"])


# ---------------------------------------------------------------------------
# 3. v1/v2 record reproduction: legacy hooked run vs the frozen tree
# ---------------------------------------------------------------------------
_SUBPROCESS_SCRIPT = r"""
import json
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, sys.argv[1])

from tensorlbm.control_volume_force import fluid_momentum_change  # noqa: F401
from tensorlbm.d3q19 import C  # noqa: F401
from tensorlbm.suboff_amr_wallmodel_runner import (  # noqa: F401
    SuboffAMRWallModelConfig,
    build_amr_contexts,
    run_suboff_amr_wallmodel,
)

config = SuboffAMRWallModelConfig(
    nx=96, ny=48, nz=48, hull_length=48.0, hull_type="BARE_HULL",
    l1_wall_margin=3, l1_wake_cells=8, cv_margin=2, steps=3, warmup_steps=0,
    ramp_steps=4, wall_diagnostic_interval=1, device="cpu",
    fine_wall_chains=False, seal_solid_mode="eroded_core",
    seal_erode_passes=3, root_bounce_seal=False,
)

built = build_amr_contexts(config, torch.device("cpu"))
ghost = built["ghost"]
context = built["contexts"][-1]
cv = context.control_volume
fluid_in_cv = cv & ~context.solid & context.physical_mask
cells = torch.nonzero(fluid_in_cv, as_tuple=False)
zc, yc, xc = (int(v) for v in cells[len(cells) // 2])
q_pos_x = int(torch.nonzero((C[:, 0] == 1) & (C[:, 1] == 0) & (C[:, 2] == 0)).flatten()[0])
DELTA = 2.0e-3


def hook(fine_old, fine_new, fine_post):
    fine_new[q_pos_x, zc - ghost, yc - ghost, xc - ghost] -= DELTA


artifact = run_suboff_amr_wallmodel(config, pre_restrict_hook=hook)
with open(sys.argv[2], "w") as fh:
    json.dump(artifact, fh, sort_keys=True)
"""


@pytest.mark.skipif(not FROZEN_TREE.exists(), reason="frozen campaign tree absent")
def test_legacy_mode_reproduces_frozen_tree_bitwise(tmp_path) -> None:
    outputs = {}
    for tag, src_root in (
        ("frozen", FROZEN_TREE / "src"),
        ("branch", Path(__file__).resolve().parent.parent / "src"),
    ):
        out = tmp_path / f"{tag}.json"
        subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(src_root), str(out)],
            check=True,
            cwd=str(tmp_path),
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        with open(out) as fh:
            outputs[tag] = json.load(fh)

    branch = _strip_additive_keys(outputs["branch"])
    assert branch == outputs["frozen"]
    # sanity: the additive keys really were present before stripping
    assert "observer_mode" in outputs["branch"]["result"]
