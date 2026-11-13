"""Unit tests for the conv-as-streaming fixed-kernel transport (XVAL m17680).

The cross-validation report lives in
``docs/XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.md``; these tests pin the three
claims it rests on:

1. the ported operator is the *same* operator as the production gather
   transport (2D bit-wise, 3D against the canonical roll traversal);
2. the 1x1 bounce convolution is the same operation as an explicit direction
   reversal;
3. the kernel is *derived* from the velocity table, and that derivation
   reproduces upstream's shipped D2Q9 table while exposing upstream's D3Q19
   defect (the self-check upstream lacks).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm import streaming_conv as SC
from tensorlbm.d2q9 import C as C2D, OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream as stream_gather

REF_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "xval_conv_as_streaming" / "oss_ref"


def _tables(name: str) -> dict:
    path = REF_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"upstream tables not extracted ({path})")
    return json.loads(path.read_text())


# --------------------------------------------------------------- equivalences
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stream_conv_2d_is_bitwise_identical_to_gather(dtype):
    torch.manual_seed(0)
    f = torch.rand(9, 17, 23, dtype=dtype)
    a, b = stream_gather(f), SC.stream_conv_2d(f)
    assert torch.equal(a, b)
    # conservation holds to round-off (the two summations use different orders)
    assert float(f.sum()) == __import__('pytest').approx(float(b.sum()), rel=1e-15)


def test_stream_conv_2d_keeps_uniform_state_uniform():
    f = equilibrium(
        torch.ones(12, 10, dtype=torch.float64),
        torch.full((12, 10), 0.03, dtype=torch.float64),
        torch.zeros(12, 10, dtype=torch.float64),
    )
    out = SC.stream_conv_2d(f)
    _, ux, _ = macroscopic(out)
    assert torch.allclose(out, f, atol=1e-15)
    assert float(ux.mean()) == pytest.approx(0.03, abs=1e-12)


def test_stream_conv_3d_matches_canonical_roll_traversal():
    from tensorlbm.core.d3q19_stencil import roll_from_pull_source

    torch.manual_seed(1)
    f = torch.rand(19, 5, 6, 7, dtype=torch.float64)
    ref = torch.empty_like(f)
    ref[0] = f[0]
    for q in range(1, 19):
        ref[q] = roll_from_pull_source(f[q], q)
    assert torch.equal(ref, SC.stream_conv_3d(f))


def test_bounce_blend_equals_explicit_reversal():
    torch.manual_seed(2)
    mask = torch.zeros(9, 11, dtype=torch.float64)
    mask[0, :] = 1.0
    mask[-1, :] = 1.0
    pre = torch.rand(9, 9, 11, dtype=torch.float64)
    post = torch.rand(9, 9, 11, dtype=torch.float64)
    blended = SC.bounce_blend(post, pre, mask, opposite=OPPOSITE)
    ref = post.clone()
    ref[:, mask > 0] = pre[OPPOSITE][:, mask > 0]
    assert torch.equal(blended, ref)


def test_collide_stream_loop_reproduces_taylor_green_decay_rate():
    """The ported transport drives the physics: analytic decay rate within 3 %."""
    torch.manual_seed(3)
    n, re, u0, steps = 32, 100.0, 0.05, 1200
    k = 2.0 * np.pi / n
    nu = u0 * n / re
    tau = 0.5 + 3.0 * nu
    jj, ii = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    x, y = ii + 0.5, jj + 0.5
    ux0 = -u0 * torch.cos(k * x) * torch.sin(k * y)
    uy0 = u0 * torch.sin(k * x) * torch.cos(k * y)
    energies = {}
    for arm, transport in (("A", stream_gather), ("B", SC.stream_conv_2d)):
        f = equilibrium(torch.ones_like(ux0), ux0, uy0)
        with torch.no_grad():
            for _ in range(steps):
                f = collide_bgk(transport(f), tau)
        _, ux, uy = macroscopic(f)
        energies[arm] = 0.5 * float((ux * ux + uy * uy).mean())
    gamma_theory = 4.0 * nu * k * k
    k0 = 0.25 * u0 * u0
    gamma = -np.log(energies["B"] / k0) / steps
    assert abs(gamma - gamma_theory) / gamma_theory < 0.03
    assert energies["A"] == pytest.approx(energies["B"], rel=1e-12)


# ----------------------------------------------------- kernel derivation audit
def test_derived_d2q9_kernel_reproduces_upstream_table():
    tab = _tables("D2Q9")
    derived = SC.conv_stream_kernel_dense(tab["lveloc"], axes="yx")
    upstream = np.asarray(tab["stream"], dtype=np.float64)
    assert np.array_equal(derived, upstream)
    assert SC.stream_table_mismatches(tab["lveloc"], upstream, "yx") == []


def test_derived_d3q15_kernel_reproduces_upstream_table():
    tab = _tables("D3Q15")
    derived = SC.conv_stream_kernel_dense(tab["lveloc"], axes="zyx")
    assert np.array_equal(derived, np.asarray(tab["stream"], dtype=np.float64))


def test_upstream_d3q19_table_defect_is_detected():
    """Upstream D3Q19 q=3 streams along a diagonal instead of +y (shipped bug)."""
    tab = _tables("D3Q19")
    bad = SC.stream_table_mismatches(tab["lveloc"], np.asarray(tab["stream"]), "zyx")
    assert len(bad) == 1
    rec = bad[0]
    assert rec["q"] == 3
    assert rec["c"] == [0, 1, 0]
    assert rec["expected"] == [1, 0, 1]
    assert rec["table"] == [0, 0, 1]


@pytest.mark.parametrize(
    ("c", "axes", "expected"),
    [
        ((1, 0, 0), "yx", (1, 0)),
        ((0, 1, 0), "yx", (0, 1)),
        ((-1, -1, 0), "yx", (2, 2)),
        ((1, 0, 0), "zyx", (1, 1, 0)),
    ],
)
def test_kernel_tap_rule(c, axes, expected):
    weight = SC.conv_stream_kernel(np.array([c]), axes=axes)
    assert tuple(int(v) for v in np.argwhere(weight[0, 0] == 1.0)[0]) == expected


def test_conv_stream_kernel_rejects_bad_velocity_table():
    with pytest.raises(ValueError):
        SC.conv_stream_kernel(np.zeros((9, 5)))