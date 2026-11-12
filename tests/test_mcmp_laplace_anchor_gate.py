"""Invariants of the MCMP-LAPLACE anchor summary (``independent_sigma._summary``).

Regression for m15969 audit finding F2: the summary used to pick the *flattest*
pairing without honouring the report's own ``sigma > 0`` gate, so on the shipped
artefact (all pairings flatness ≈ 0.44, ``bwd`` sigma ≡ 0) the documented repro
command died with ``ZeroDivisionError`` at ``rel_err_pct``.

The tests feed synthetic level dicts -- the gate logic is what is under test, the
numbers are irrelevant -- and assert the summary's contract:
no division by a zero sigma, and a *silent* fallback is not allowed: the level
must be flagged rather than scored.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "case_pool" / "mcmp_laplace" / "independent_sigma.py"


def _load():
    spec = importlib.util.spec_from_file_location("_indep_sigma", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pairings(*, sigmas: dict[str, float], flatness: float) -> dict[str, dict]:
    return {
        p: {"sigma": sigmas[p], "flatness": flatness}
        for p in ("fwd", "bwd", "sym", "isotropic")
    }


def _level(pairings: dict[str, dict], *, radius: float = 15.0, dp: float = 0.0135) -> dict:
    return {
        "G12": -2.5,
        "flat": {
            "96": {"pairings": {p: dict(v) for p, v in pairings.items()}},
            "128": {"pairings": {p: dict(v) for p, v in pairings.items()}},
        },
        "droplets": [
            {
                "n": 128,
                "radius": radius,
                "delta_p": dp,
                "sigma_laplace": dp * radius,
                "converged": True,
            }
        ],
    }


def test_all_pairings_vetoed_by_gate_does_not_divide_by_zero():
    """Shipped-artefact shape: sigma <= 0 everywhere, flatness ~0.44."""
    mod = _load()
    res = {
        "levels": [
            _level(_pairings(sigmas={"fwd": -0.0201, "bwd": 0.0, "sym": -0.0101, "isotropic": 0.0},
                             flatness=0.436))
        ]
    }
    out = mod._summary(res)  # used to raise ZeroDivisionError
    assert out["rows"] == []
    assert out["gate_failed"] == {"G-2.5": True}
    assert out["best_pairing"] == {"G-2.5": None}


def test_positive_sigma_pairing_is_selected_and_finite():
    mod = _load()
    pairings = _pairings(sigmas={"fwd": 0.30, "bwd": 0.0, "sym": 0.15, "isotropic": 0.0},
                         flatness=0.03)
    pairings["fwd"]["flatness"] = 0.045  # flattest eligible pairing is ``sym``
    pairings["sym"]["flatness"] = 0.020
    pairings["bwd"]["flatness"] = 0.060  # fails the 5% normal-force gate anyway
    res = {"levels": [_level(pairings)]}
    out = mod._summary(res)
    assert out["gate_failed"] == {"G-2.5": False}
    assert out["best_pairing"] == {"G-2.5": "sym"}  # flattest pairing with sigma > 0
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    assert math.isfinite(row["rel_err_pct"])
    assert row["dp_ref"] == pytest.approx(0.15 / 15.0)


def test_flat_but_zero_sigma_pairing_is_not_selected():
    """bwd is the flattest but sigma == 0: it must not win the selection."""
    mod = _load()
    pairings = _pairings(sigmas={"fwd": 0.30, "bwd": 0.0, "sym": 0.15, "isotropic": 0.0},
                         flatness=0.03)
    pairings["sym"]["flatness"] = 0.020
    pairings["bwd"]["flatness"] = 0.001
    out = mod._summary({"levels": [_level(pairings)]})
    assert out["best_pairing"] == {"G-2.5": "sym"}