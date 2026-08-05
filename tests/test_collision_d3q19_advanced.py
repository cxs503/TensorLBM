"""Contract tests for the reusable, honestly scoped D3Q19 stress kernels."""
import importlib.util
import subprocess
from pathlib import Path

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d


def _state():
    torch.manual_seed(19)
    rho = 0.9 + torch.rand(2, 3, 4)
    ux = 0.03 * torch.randn_like(rho)
    uy = 0.03 * torch.randn_like(rho)
    uz = 0.03 * torch.randn_like(rho)
    feq = equilibrium3d(rho, ux, uy, uz)
    # A non-hydrodynamic perturbation makes the regularizing projection observable.
    return feq + 1.0e-3 * torch.randn_like(feq)


@pytest.fixture(scope="module")
def legacy_cg_module():
    """Load the base revision under a separate module name for exact replay."""
    source = subprocess.check_output(
        ["git", "show", "10ac615daf0623685be217a197eabac6b7cf3786:src/tensorlbm/cg_advanced_collision.py"],
        text=True,
    )
    path = Path(__file__).with_name("_legacy_cg_advanced_collision.py")
    path.write_text(source)
    try:
        spec = importlib.util.spec_from_file_location(
            "tensorlbm._legacy_cg_advanced_collision", path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load base CG collision module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("cumulant", {}),
        ("cascaded", {}),
        ("cascaded", {"s_bulk": 1.31}),
        ("kbc", {}),
        ("kbc", {"C_s": 0.22}),
    ],
)
def test_legacy_cg_public_apis_are_bitwise_replays_of_base(dtype, name, kwargs, legacy_cg_module):
    """Common extraction must retain base's fixed-float32 lattice arithmetic."""
    from tensorlbm import cg_advanced_collision as extracted

    torch.manual_seed(817)
    f = torch.rand((19, 2, 3, 4), dtype=dtype) + 0.02
    red, blue = 0.61 * f, 0.39 * f
    call_kwargs = dict(tau=0.87, A=0.017, beta=0.63, gx=0.0012, gy=-0.0008, gz=0.0004)
    call_kwargs.update(kwargs)
    baseline = getattr(legacy_cg_module, f"collide_cg_{name}_3d")(red.clone(), blue.clone(), **call_kwargs)
    with pytest.warns(DeprecationWarning, match="WITHHELD"):
        actual = getattr(extracted, f"collide_cg_{name}_3d")(red.clone(), blue.clone(), **call_kwargs)
    assert torch.equal(actual[0], baseline[0])
    assert torch.equal(actual[1], baseline[1])


def test_stress_projection_reconstructs_its_six_second_moments():
    from tensorlbm.collision_d3q19_advanced import (
        reconstruct_second_order_stress_d3q19,
        second_order_stress_d3q19,
    )

    fneq = _state() - equilibrium3d(*macroscopic3d(_state()))
    stress = second_order_stress_d3q19(fneq)
    projected = reconstruct_second_order_stress_d3q19(*stress)
    reconstructed_stress = second_order_stress_d3q19(projected)
    for actual, expected in zip(reconstructed_stress, stress):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)


def test_regularized_stress_kernel_accepts_caller_equilibrium_and_fixes_equilibrium():
    from tensorlbm.collision_d3q19_advanced import collide_regularized_stress_d3q19

    f = _state()
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    post = collide_regularized_stress_d3q19(f, feq, tau=0.83)
    rho_post, ux_post, uy_post, uz_post = macroscopic3d(post)
    torch.testing.assert_close(rho_post, rho, atol=3e-7, rtol=3e-7)
    torch.testing.assert_close(ux_post, ux, atol=3e-7, rtol=3e-7)
    torch.testing.assert_close(uy_post, uy, atol=3e-7, rtol=3e-7)
    torch.testing.assert_close(uz_post, uz, atol=3e-7, rtol=3e-7)
    torch.testing.assert_close(collide_regularized_stress_d3q19(feq, feq, tau=0.83), feq)


def test_common_stress_kernels_use_the_exact_caller_provided_equilibrium():
    from tensorlbm.collision_d3q19_advanced import (
        collide_central_stress_d3q19,
        collide_regularized_stress_d3q19,
    )

    f_total = _state()
    rho, ux, uy, uz = macroscopic3d(f_total)
    shifted_feq = equilibrium3d(rho, ux + 0.015, uy - 0.01, uz + 0.005)
    for kernel, kwargs in (
        (collide_regularized_stress_d3q19, {}),
        (collide_central_stress_d3q19, {"s_bulk": 1.3}),
    ):
        expected = kernel(f_total, shifted_feq, tau=0.83, **kwargs)
        assert expected.shape == f_total.shape
        torch.testing.assert_close(kernel(shifted_feq, shifted_feq, tau=0.83, **kwargs), shifted_feq)


def test_common_mrt_name_delegates_to_unchanged_solver3d_baseline():
    from tensorlbm.collision_d3q19_advanced import collide_mrt_d3q19
    from tensorlbm.solver3d import collide_mrt3d

    f = _state()
    torch.testing.assert_close(collide_mrt_d3q19(f, 0.91), collide_mrt3d(f, 0.91))


def test_cg_legacy_names_warn_and_are_explicitly_withheld_not_kbc_or_cascaded():
    from tensorlbm.cg_advanced_collision import collide_cg_cascaded_3d, collide_cg_kbc_3d

    f = _state()
    with pytest.warns(DeprecationWarning, match="WITHHELD"):
        cascaded = collide_cg_cascaded_3d(0.6 * f, 0.4 * f, tau=0.9)
    with pytest.warns(DeprecationWarning, match="WITHHELD"):
        kbc = collide_cg_kbc_3d(0.6 * f, 0.4 * f, tau=0.9)
    assert cascaded[0].shape == f.shape
    assert kbc[0].shape == f.shape


def test_cg_private_reconstruction_keeps_legacy_seventh_positional_device():
    from tensorlbm.cg_advanced_collision import _reconstruct_fneq_d3q19
    from tensorlbm.collision_d3q19_advanced import (
        reconstruct_second_order_stress_d3q19,
        second_order_stress_d3q19,
    )

    fneq = _state() - equilibrium3d(*macroscopic3d(_state()))
    stress = second_order_stress_d3q19(fneq)
    expected = reconstruct_second_order_stress_d3q19(*stress)
    actual = _reconstruct_fneq_d3q19(
        stress[0], stress[1], stress[2], stress[3], stress[4], stress[5], fneq.device
    )
    torch.testing.assert_close(actual, expected)
