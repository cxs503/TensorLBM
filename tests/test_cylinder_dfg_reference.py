"""Guards for the Schäfer-Turek DFG 2D-2 case (mission m15776).

The point of these tests is *not* to re-run the benchmark (that is the case
runner's job) but to freeze the three things that silently rot:

1. the reference bands still match the frozen register (zero hand-copy),
2. the pressure probes never land on solid cells,
3. the Strouhal extraction recovers a known period from a synthetic signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "validation_matrix"))

from tensorlbm.boundaries import cylinder_mask  # noqa: E402
from tensorlbm.validation import cylinder_dfg as cdfg  # noqa: E402

MATRIX_REFS = (
    "cylinder_dfg2d2_cd_re100",
    "cylinder_dfg2d2_clamp_re100",
    "cylinder_dfg2d2_st_re100",
    "cylinder_dfg2d2_dp_re100",
)


def test_reference_anchors_audit_clean():
    """The bands typed in the module agree with the frozen REFERENCE.md."""
    import reference_audit

    refs = reference_audit.load_references()
    for name in MATRIX_REFS:
        rec = reference_audit.audit_reference(name, refs[name])
        assert rec["pass"], rec["reasons"]
    src = (REPO / "benchmarks/verified/cylinder/re100_dfg/REFERENCE.md").read_text()
    for q, band in (("cd", (3.22, 3.24)), ("st", (0.295, 0.305)),
                    ("cl_amp", (0.99, 1.01)), ("dp", (2.46, 2.50))):
        assert cdfg.REFERENCE_BANDS["2D-2"][q] == band, q
        assert f"{band[0]:.4f}" in src and f"{band[1]:.4f}" in src, q


@pytest.mark.parametrize("D", [20, 28, 40])
def test_probe_columns_are_fluid(D):
    cfg = cdfg.CaseConfig(D_cells=D)
    pr = cdfg.probe_columns(cfg)
    mask = cylinder_mask(cfg.nx, cfg.ny, round(cdfg.CYL_CX / cfg.dx),
                         round(cdfg.CYL_CY / cfg.dx), 0.5 * cdfg.CYL_D / cfg.dx,
                         torch.device("cpu"))
    for key in ("up_near", "up_far", "dn_near", "dn_far"):
        assert not bool(mask[pr["row"], pr[key]]), (D, key)
    # the DFG probe x positions themselves sit on the digital cylinder surface
    assert bool(mask[pr["row"], int(round(cdfg.PROBE_A[0] / cfg.dx))])
    assert bool(mask[pr["row"], int(round(cdfg.PROBE_B[0] / cfg.dx))])


def test_strouhal_extraction_recovers_known_period():
    """Synthetic lift signal: D=20, u=0.0833, St=0.30 -> period 800 steps."""
    rng = np.random.default_rng(7)
    n = 20_000
    period = 20 / (0.30 * 0.0833)
    t = np.arange(n)
    cl = 1.0 * np.sin(2 * np.pi * t / period) + 0.02 * rng.standard_normal(n)
    out = cdfg.strouhal_from_series(torch.tensor(cl, dtype=torch.float32), period, 0.30, 20, 0.0833)
    assert abs(out["st_acf"] - 0.30) / 0.30 < 0.02
    assert abs(out["st_fft"] - 0.30) / 0.30 < 0.02


def test_smoke_run_is_finite():
    """Tiny grid, few steps: the entry path executes and returns finite numbers."""
    cfg = cdfg.CaseConfig(D_cells=10, n_steps=200, avg_periods=1.0, report_every=0,
                          row_sample_every=1, seed_ampl=0.02)
    res = cdfg.run_case(cfg, threads=2, keep_series=True)
    assert np.isfinite(res["cd"]) and res["cd"] > 0
    assert np.isfinite(res["dp"])
    assert len(res["series"]["cl"]) == res["avg_window"]
    assert res["n_solid_cells"] > 0.7 * 3.14159 * (0.5 * cfg.D_cells) ** 2
