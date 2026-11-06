"""Cold-path guard tests for :mod:`tensorlbm.backends.triton_backend`.

The Triton step itself needs a CUDA device, so on a CPU CI host every numerics
path is WITHHELD.  What *is* testable — and what previously had zero coverage —
is the whole cold path around it:

* ``is_available()`` must report exactly the documented predicate, otherwise a
  runner will try to launch kernels on a host without CUDA;
* :class:`TritonStepState` buffer layout (ping-pong pair must not alias, Q=19
  production layout, lazily cached equilibrium vector);
* the argument guards in :func:`triton_suboff_step`, which must reject bad
  input *before* any kernel launch.  To reach the guards that sit behind the
  CUDA check we use a ``torch.Tensor`` subclass that reports ``is_cuda=True``
  while living on the CPU — the guard only reads meta-data, so no kernel is
  ever compiled.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.backends import triton_backend as tb
from tensorlbm.core.lattice import D3Q19

N = 4


class _CudaClaimingTensor(torch.Tensor):
    """CPU tensor that lies about ``is_cuda`` to exercise meta-data guards."""

    is_cuda = property(lambda self: True)  # noqa: ARG005


def _cuda_claiming(shape: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
    base = torch.full(shape, value, dtype=torch.float32)
    return _CudaClaimingTensor(base)


def _wall(shape: tuple[int, ...] = (N, N, N)) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int8)


def _step(f: torch.Tensor, obstacle: torch.Tensor, **kwargs: object):
    params: dict[str, object] = {
        "u_in": 0.05,
        "step": 1,
        "mass_period": 10,
        "initial_mass": 1.0,
        "nu_lb": 0.01,
    }
    params.update(kwargs)
    return tb.triton_suboff_step(f, obstacle, **params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Availability + constants (positive)
# ---------------------------------------------------------------------------


def test_production_q_matches_d3q19() -> None:
    assert tb._PROD_Q == D3Q19.q == 19


def test_is_available_reports_the_documented_predicate() -> None:
    assert tb.is_available() is (torch.cuda.is_available() and tb._Q_PAD > 0)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA host: predicate cannot be False")
def test_is_available_is_false_on_cpu_host() -> None:
    assert tb.is_available() is False


def test_public_surface_is_declared() -> None:
    assert tb.__all__ == ["TritonStepState", "triton_suboff_step", "is_available"]


# ---------------------------------------------------------------------------
# TritonStepState buffer contract (positive)
# ---------------------------------------------------------------------------


def test_step_state_allocates_production_layout() -> None:
    f = torch.zeros((19, N, N, N), dtype=torch.float32)
    state = tb.TritonStepState(f, torch.device("cpu"))
    assert state.f_buf.shape == (19, N, N, N)
    assert state.f_buf_alt.shape == (19, N, N, N)
    assert state.f_buf.dtype is torch.float32
    assert state.f_buf_alt.dtype is torch.float32
    # Ping-pong halves must be distinct allocations: streaming out of the
    # buffer being written silently doubles drag.
    assert state.f_buf.data_ptr() != state.f_buf_alt.data_ptr()
    assert state.f_buf.data_ptr() != f.data_ptr()
    assert state.f_buf_alt.data_ptr() != f.data_ptr()


def test_step_state_starts_without_cached_equilibrium_and_zeroed_scratch() -> None:
    f = torch.ones((19, N, N, N), dtype=torch.float32)
    state = tb.TritonStepState(f, torch.device("cpu"))
    assert state.feq_vec_buf is None
    assert state.feq_vec_u is None
    assert state.feq_pad_buf.shape == (19,)
    assert float(state.feq_pad_buf.abs().sum()) == 0.0
    assert state.tau_eff_buf.shape == (N, N, N)
    assert state.tau_eff_buf.dtype is torch.float32
    for force_buf in (state.fx_buf, state.fy_buf, state.fz_buf):
        assert force_buf.shape == ()
        assert force_buf.dtype is torch.float32
        assert float(force_buf) == 0.0


def test_step_state_uses_slots() -> None:
    state = tb.TritonStepState(torch.zeros((19, 2, 2, 2)), torch.device("cpu"))
    assert not hasattr(state, "__dict__")
    assert set(state.__slots__) >= {"f_buf", "f_buf_alt", "feq_vec_buf", "tau_eff_buf"}


def test_step_state_accepts_non_cubic_spatial_shape() -> None:
    state = tb.TritonStepState(torch.zeros((19, 3, 5, 7)), torch.device("cpu"))
    assert state.f_buf.shape == (19, 3, 5, 7)
    assert state.tau_eff_buf.shape == (3, 5, 7)


# ---------------------------------------------------------------------------
# Argument guards (negative)
# ---------------------------------------------------------------------------


def test_cpu_tensor_is_rejected_before_anything_runs() -> None:
    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        _step(torch.zeros((19, N, N, N)), _wall())


def test_wrong_q_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match production D3Q19"):
        _step(_cuda_claiming((27, N, N, N)), _wall())


def test_q_is_checked_before_obstacle_shape() -> None:
    """Guard order is part of the contract: Q first, then mask shape."""
    with pytest.raises(ValueError, match="production D3Q19"):
        _step(_cuda_claiming((27, N, N, N)), _wall((N, N)))


def test_obstacle_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="obstacle_int8 shape"):
        _step(_cuda_claiming((19, N, N, N)), _wall((N, N)))


def test_tau_eff_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="tau_eff shape"):
        _step(_cuda_claiming((19, N, N, N)), _wall(), tau_eff=torch.zeros((N, N)))


def test_valid_shapes_pass_the_meta_data_guards() -> None:
    """With correct Q/shapes the guards fall through to the kernel launch.

    No CUDA device is present, so the launch (or the Triton JIT) is what fails
    — and crucially *not* one of the wrapper's own ``ValueError``/``RuntimeError``
    meta-data guards.  This is the only positive-path assertion available on a
    CPU host; the numerics remain WITHHELD without a GPU.
    """
    f = _cuda_claiming((19, N, N, N))
    obstacle = _wall()
    tau_eff = torch.zeros((N, N, N), dtype=torch.float32)
    with pytest.raises(Exception) as excinfo:
        _step(f, obstacle, tau_eff=tau_eff)
    message = str(excinfo.value)
    assert "does not match" not in message
    assert "requires a CUDA tensor" not in message
