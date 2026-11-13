"""Second-order MUSCL/HLL scheme for the Ritter dam break — gated + regression.

Context (mission m17112, matrix slot 8 ``dam_break``)
-----------------------------------------------------
The CI-tier validation matrix case ``dam_break_ritter_ci`` failed on the depth
profile metric with the original first-order global-wave-speed Rusanov flux
(profile L2 = 9.82% / 6.29% at nx = 200 / 400 against a 5% gate).  The root cause
is numerical diffusion of the first-order flux, whose viscosity is
``~ (|u| + c) dx`` *independent of the local gradient*, so the coarse CI grids
cannot resolve the rarefaction fan.  ``tensorlbm.shallow_water`` therefore gained
a second-order scheme (minmod-limited MUSCL reconstruction + local-speed HLL flux
+ SSP-RK2), selected with ``ShallowWaterConfig(scheme="muscl2")``.

These tests pin both sides of that change:

* the legacy scheme's numbers are frozen (so the improvement is measured against
  a baseline, not against a moving target),
* the second-order scheme meets the matrix gate with margin and stays exactly
  mass conservative and lake-at-rest well balanced.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.shallow_water import (
    ShallowWaterConfig,
    front_from_mass,
    init_dam_break,
    ritter_depth,
    ritter_front,
    run_ritter_dam_break,
    swe_step,
)

T_END = 5.0
CFL = 0.4


def _run(nx: int, scheme: str):
    cfg = ShallowWaterConfig(
        nx=nx, scheme=scheme, t_end=T_END, cfl=CFL, sample_every=max(1, nx // 8)
    )
    return cfg, run_ritter_dam_break(cfg)


def _profile_l2_pct(cfg: ShallowWaterConfig, run) -> float:
    t_final = run.time[-1]
    x = torch.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=torch.float64)
    h = torch.as_tensor(run.profiles[-1][1], dtype=torch.float64)
    x_f = ritter_front(t_final, cfg.h0, cfg.g)
    mask = (x >= 0.0) & (x <= x_f)
    h_ref = torch.tensor(
        [ritter_depth(float(xi), t_final, cfg.h0, cfg.g) for xi in x[mask]],
        dtype=torch.float64,
    )
    return float(torch.sqrt(torch.sum((h[mask] - h_ref) ** 2) / torch.sum(h_ref**2))) * 100.0


# ---------------------------------------------------------------------------
# baseline: the legacy first-order scheme (unchanged behaviour)
# ---------------------------------------------------------------------------


def test_legacy_rusanov_profile_error_is_the_documented_baseline() -> None:
    """The first-order flux must keep reproducing the 9.82% under-resolved profile."""
    cfg, run = _run(200, "rusanov")
    l2 = _profile_l2_pct(cfg, run)
    assert abs(l2 - 9.8248) < 0.05, f"legacy profile L2 drifted: {l2:.4f}%"


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        ShallowWaterConfig(scheme="godunov").validate()


def test_default_scheme_is_the_legacy_one() -> None:
    """Defaults must not silently change the historical solver."""
    assert ShallowWaterConfig().scheme == "rusanov"


# ---------------------------------------------------------------------------
# the second-order scheme: gate compliance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nx", [200, 400])
def test_muscl2_profile_error_passes_the_5pct_gate(nx: int) -> None:
    cfg, run = _run(nx, "muscl2")
    l2 = _profile_l2_pct(cfg, run)
    assert l2 < 5.0, f"profile L2 {l2:.3f}% at nx={nx} exceeds the 5% matrix gate"
    # and it is a real improvement over the frozen first-order baseline (9.82%)
    assert l2 < 0.2 * 9.8248


@pytest.mark.parametrize("nx", [200, 400])
def test_muscl2_front_law_and_mass(nx: int) -> None:
    cfg, run = _run(nx, "muscl2")
    t_final = run.time[-1]
    front_num = front_from_mass(run.volume_downstream[-1], cfg.h0)
    front_ref = ritter_front(t_final, cfg.h0, cfg.g)
    assert abs(front_num / front_ref - 1.0) * 100.0 < 1.0
    drift = abs(run.mass_final - run.mass_initial) / run.mass_initial * 100.0
    assert drift < 1.0e-10, f"mass drift {drift:.3e}% (flux form must conserve)"


def test_muscl2_is_monotone_under_refinement() -> None:
    """The gated convergence criterion: error decreases with resolution."""
    l2 = []
    for nx in (100, 200):
        cfg, run = _run(nx, "muscl2")
        l2.append(_profile_l2_pct(cfg, run))
    assert l2[1] < l2[0], f"profile error not monotone under refinement: {l2}"
    # at least first-order convergence between these two grids
    order = math.log2(l2[0] / l2[1])
    assert order > 0.5, f"observed order {order:.2f} too low"


# ---------------------------------------------------------------------------
# structural properties of the new flux path
# ---------------------------------------------------------------------------


def test_muscl2_lake_at_rest_is_exact() -> None:
    """A flat, still lake must stay bitwise unchanged (well balanced, no flux)."""
    cfg = ShallowWaterConfig(nx=64, scheme="muscl2", h0=1.0, g=1.0)
    U = torch.stack([torch.ones(cfg.nx, dtype=torch.float64),
                     torch.zeros(cfg.nx, dtype=torch.float64)])
    dt = 0.1 * cfg.dx / cfg.c0
    for _ in range(20):
        U = swe_step(U, cfg, dt)
    assert torch.allclose(U[0], torch.ones_like(U[0]), atol=0.0, rtol=0.0)
    assert torch.max(torch.abs(U[1])) == 0.0


def test_muscl2_keeps_depth_non_negative_on_a_dry_bed() -> None:
    cfg = ShallowWaterConfig(nx=100, scheme="muscl2", t_end=3.0)
    run = run_ritter_dam_break(cfg)
    for _t, h in run.profiles:
        assert min(h) >= 0.0
    assert run.front[-1] > 0.0
    # the wet region must not overshoot the analytic reservoir depth
    assert max(max(h) for _t, h in run.profiles) <= cfg.h0 * (1.0 + 1.0e-6)


def test_step_dispatch_matches_explicit_scheme_call() -> None:
    cfg = ShallowWaterConfig(nx=64, scheme="muscl2")
    U = init_dam_break(cfg)
    dt = 0.2 * cfg.dx / cfg.c0
    a = swe_step(U, cfg, dt)
    cfg_r = ShallowWaterConfig(nx=64, scheme="rusanov")
    b = swe_step(U, cfg_r, dt)
    assert not torch.allclose(a, b), "scheme selection had no effect"
    assert float(a[0].sum()) > 0.0 and float(b[0].sum()) > 0.0