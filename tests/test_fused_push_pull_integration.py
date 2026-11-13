"""Integration of the fused push-pull kernel into the library stepping path.

Anchor ``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN``.  The first delivery of this
anchor (mission 17839) shipped ``tensorlbm.fused_push_pull`` as a stand-alone
module with no caller; the independent audit recorded that as the reason the
capability could not be graded above L2.  These tests pin the caller
(:func:`tensorlbm.solver.step_stream_collide`) and, more importantly, the exact
relation between the two kernels, because "bit-identical to the AB reference"
is *false* if you compare the raw states:

* the fused kernel collides **after** streaming, the AB reference **before**;
* at the same step count, ``f_fused == collide_bgk(f_ab)`` to round-off;
* the raw ``max|f_fused - f_ab|`` is that one-collision label offset, O(Ma),
  and must not be mistaken for an error.
"""

import math

import pytest
import torch

from tensorlbm.d2q9 import C
from tensorlbm.fused_push_pull import equilibrium_upstream, taylor_green_fields
from tensorlbm.solver import STEP_KERNELS, collide_bgk, step_stream_collide

NU = 1.0 / 6.0
TAU = 3.0 * NU + 0.5
U_MAX = 0.02


def _initial(n: int = 24) -> torch.Tensor:
    rho, ux, uy = taylor_green_fields(n, n, u_max=U_MAX, nu=NU, t=0)
    return equilibrium_upstream(rho, ux, uy)


def _momenta(f: torch.Tensor):
    rho = f.sum(0)
    c = C.to(f.device)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(0) / rho
    uy = (f * c[:, 1].view(9, 1, 1)).sum(0) / rho
    return rho, ux, uy


def _run(f: torch.Tensor, steps: int, kernel: str, tau: float = TAU) -> torch.Tensor:
    for _ in range(steps):
        f = step_stream_collide(f, tau, kernel=kernel)
    return f


def test_kernels_are_registered():
    assert STEP_KERNELS == ("ab", "fused_push_pull")


def test_fused_is_the_ab_step_with_collision_at_the_end():
    """f_fused(n) == collide_bgk(f_ab(n)) to round-off (states compared same-level)."""
    f0 = _initial()
    f_ab = _run(f0.clone(), 100, "ab")
    f_fu = _run(f0.clone(), 100, "fused_push_pull")
    diff = float((f_fu - collide_bgk(f_ab, TAU)).abs().max())
    assert diff < 1e-12, diff
    # and the raw state difference is the one-collision offset, not round-off
    assert 1e-8 < float((f_fu - f_ab).abs().max()) < 1.0


def test_velocity_fields_agree_between_kernels():
    f0 = _initial()
    f_ab = _run(f0.clone(), 100, "ab")
    f_fu = _run(f0.clone(), 100, "fused_push_pull")
    _, ux_a, uy_a = _momenta(f_ab)
    _, ux_b, uy_b = _momenta(f_fu)
    num = float(((ux_a - ux_b) ** 2 + (uy_a - uy_b) ** 2).sum()) ** 0.5
    den = float((ux_a**2 + uy_a**2).sum()) ** 0.5
    assert num / den < 1e-10, num / den


@pytest.mark.parametrize("kernel", ["ab", "fused_push_pull"])
def test_mass_and_momentum_conservation(kernel):
    f0 = _initial()
    mass0 = float(f0.sum())
    px0 = float((f0 * C[:, 0].view(9, 1, 1)).sum())
    py0 = float((f0 * C[:, 1].view(9, 1, 1)).sum())
    f = _run(f0, 40, kernel)
    assert abs(float(f.sum()) - mass0) / mass0 < 1e-13
    assert abs(float((f * C[:, 0].view(9, 1, 1)).sum()) - px0) < 1e-11
    assert abs(float((f * C[:, 1].view(9, 1, 1)).sum()) - py0) < 1e-11


@pytest.mark.parametrize("kernel", ["ab", "fused_push_pull"])
def test_physics_gate_taylor_green(kernel):
    """Closed-form Taylor-Green decay: u_x relative L2 stays inside the 3% target."""
    n, steps = 32, 120
    f0 = _initial(n)
    f = _run(f0, steps, kernel)
    rho, ux, uy = _momenta(f)
    ra, uxa, uya = taylor_green_fields(n, n, u_max=U_MAX, nu=NU, t=steps)
    err = float(((ux - uxa) ** 2).sum()) ** 0.5 / float((uxa**2).sum()) ** 0.5
    assert err < 0.03, err
    assert float(rho.sum()) - n * n < 1e-10 * n * n


def test_unknown_kernel_is_rejected():
    f0 = _initial(8)
    with pytest.raises(ValueError, match="unknown kernel"):
        step_stream_collide(f0, TAU, kernel="pull_aa")


@pytest.mark.parametrize("kernel", ["ab", "fused_push_pull"])
def test_illegal_relaxation_is_rejected(kernel):
    f0 = _initial(8)
    with pytest.raises(Exception):
        step_stream_collide(f0, 0.5, kernel=kernel)


def test_fused_converges_to_the_same_analytic_solution():
    """Two-grid ladder: the fused kernel is second-order accurate on its own."""
    errs = []
    for n, steps in ((16, 30), (32, 120)):
        f = _run(_initial(n), steps, "fused_push_pull")
        _, ux, uy = _momenta(f)
        _, uxa, uya = taylor_green_fields(n, n, u_max=U_MAX, nu=NU, t=steps)
        errs.append(float(((ux - uxa) ** 2).sum()) ** 0.5 / float((uxa**2).sum()) ** 0.5)
    order = math.log(errs[0] / errs[1]) / math.log(2.0)
    assert order > 1.0, (errs, order)


def test_verified_benchmark_quick_mode_passes(tmp_path):
    """The registered referee (run.py --quick + verify.py) must exit 0."""
    import subprocess
    import sys

    import os

    repo = pathlib.Path(__file__).resolve().parents[1]
    run = repo / "benchmarks" / "verified" / "fused_push_pull_aa_pattern"
    # The subprocess must resolve *this* worktree's `src/` (an editable install of
    # another checkout may be on site-packages), so pin PYTHONPATH explicitly.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(
        [sys.executable, str(run / "run.py"), "--quick", "--out", str(tmp_path / "result.json")],
        check=True,
        capture_output=True,
        timeout=600,
        env=env,
    )
    out = subprocess.run(
        [sys.executable, str(run / "verify.py"), "--result", str(tmp_path / "result.json")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "VERDICT PASS" in out.stdout


import pathlib  # noqa: E402  (used by the subprocess test above)
