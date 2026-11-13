"""Tests for the TCLB per-margin-arrays / pointer-shift streaming port (XVAL m18129).

The cross-validation report is
``docs/XVAL-LBM-PER-MARGIN-ARRAYS-POINTER-SHIFT-STREAMING.md``.  These tests pin
the three claims it rests on:

1. the ported transport is the *same* operator as the production gather
   transport (bit-wise, 2D and 3D);
2. the per-margin block decomposition is a *partition* of the destination cells
   and the flat offset inside each block is a *constant* -- the property that
   licenses replacing per-cell index arithmetic by a moved base pointer in the
   OSS implementation;
3. the block count per direction follows the stencil (``2**|{a: c_a != 0}|``),
   i.e. the decomposition is derived from the velocity table, not hand-written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorlbm import solver, solver3d
from tensorlbm.d2q9 import C as C2D
from tensorlbm.d2q9 import equilibrium
from tensorlbm.d3q19 import C as C3D
from tensorlbm.xval import per_margin_pointer_shift as PMPS

OSS_REF = Path(__file__).resolve().parents[1] / "benchmarks" / "xval_tclb_pointer_shift" / "oss_ref" / "REF.json"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(17, 23), (4, 4), (3, 5)])
def test_stream_2d_is_bitwise_identical_to_gather(dtype, shape):
    torch.manual_seed(0)
    f = torch.rand(9, *shape, dtype=dtype)
    plan = PMPS.pointer_shift_plan(C2D, shape)
    a = solver.stream(f)
    b = PMPS.stream_pointer_shift(f, C2D, plan=plan)
    c = PMPS.stream_pointer_shift_flat(f, C2D, plan=plan)
    assert torch.equal(a, b)
    assert torch.equal(a, c)
    tol = 1e-6 if dtype is torch.float32 else 1e-12
    assert float(f.sum()) == pytest.approx(float(b.sum()), rel=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(6, 7, 8), (5, 5, 5), (3, 4, 5)])
def test_stream_3d_is_bitwise_identical_to_gather(dtype, shape):
    torch.manual_seed(1)
    f = torch.rand(19, *shape, dtype=dtype)
    plan = PMPS.pointer_shift_plan(C3D, shape)
    a = solver3d.stream3d_gather(f)
    b = PMPS.stream_pointer_shift(f, C3D, plan=plan)
    assert torch.equal(a, b)


def test_plan_is_a_partition_with_constant_block_offsets():
    torch.manual_seed(2)
    for lveloc, shape in ((C2D, (7, 9)), (C3D, (5, 6, 7))):
        f = torch.rand(len(lveloc), *shape, dtype=torch.float64)
        inv = PMPS.check_plan_invariant(f, lveloc)
        assert inv["partition_ok"], inv
        assert inv["holes"] == 0 and inv["overlaps"] == 0
        assert inv["max_offset_error"] == 0, inv
        assert inv["max_abs_diff_vs_gather"] == 0.0, inv
        assert inv["cells_covered"] == inv["cells_expected"]


def test_block_count_follows_the_stencil():
    for lveloc, dims in ((C2D, 2), (C3D, 3)):
        plan = PMPS.pointer_shift_plan(lveloc, (8,) * dims)
        per_dir = plan.summary()["blocks_per_direction"]
        for q, row in enumerate(lveloc.tolist()):
            expected = 2 ** sum(1 for c in row[:dims] if int(c) != 0)
            assert per_dir[q] == expected, (q, per_dir[q], expected)


def test_short_axis_is_rejected():
    with pytest.raises(ValueError):
        PMPS.pointer_shift_plan(C2D, (2, 8))


def test_uniform_equilibrium_stays_uniform():
    rho = torch.ones(12, 10, dtype=torch.float64)
    ux = torch.full((12, 10), 0.03, dtype=torch.float64)
    f = equilibrium(rho, ux, torch.zeros(12, 10, dtype=torch.float64))
    out = PMPS.stream_pointer_shift(f, C2D)
    assert torch.allclose(out, f, atol=1e-15)


def test_oss_reference_is_pinned():
    if not OSS_REF.exists():
        pytest.skip("run benchmarks/xval_tclb_pointer_shift/extract_oss_ref.py first")
    ref = json.loads(OSS_REF.read_text())
    assert ref["oss"] == "CFD-GO/TCLB"
    assert len(ref["commit"]) == 40
    assert ref["tclb_runnable_here"] is False
    keys = {s["key"] for s in ref["snippets"]}
    assert {"ftabs_per_margin_arrays", "offsets_block_addressing", "move_pocket_pointer_shift"} <= keys
