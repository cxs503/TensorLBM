"""Instruction-DSL tests (grammar, invariants, lowering, physics anchors).

The tests are the deliverable of ``XVAL-LBM-INSTRUCTION-DSL-TESTS-SH``: every
declared instruction program must either be rejected with a *specific* diagnostic
or execute with its invariants verified numerically.  Physics assertions carry
the owner tolerance (>15 % unacceptable) and additionally a strict 3 % band; the
numbers are produced by the DSL itself, not hand-entered.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.instruction_dsl import (
    D2Q9_EQUILIBRIA,
    D2Q9_POLYNOMIALS,
    D2Q9_SCHEME_INSTRUCTIONS,
    DSLError,
    DSLSemanticError,
    DSLSyntaxError,
    build_scheme,
    decay_rate_from_history,
    parse,
    poiseuille_error,
    run,
    taylor_green_velocity,
)

SCHEME = D2Q9_SCHEME_INSTRUCTIONS


def program(body: str) -> str:
    return SCHEME + body


UNIFORM_BODY = "domain nx=8 ny=8\ninit uniform\nrun steps=1\n"


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------
def test_parse_canonical_program():
    prog = parse(program(UNIFORM_BODY))
    assert prog.dim == 2
    assert prog.velocity == 1.0
    assert len(prog.velocities) == 9
    assert prog.polynomials == list(D2Q9_POLYNOMIALS)
    assert prog.equilibria == list(D2Q9_EQUILIBRIA)
    assert (prog.nx, prog.ny) == (8, 8)
    assert prog.boundary == {
        "xmin": "periodic",
        "xmax": "periodic",
        "ymin": "periodic",
        "ymax": "periodic",
    }
    assert prog.steps == 1


def test_digest_is_content_addressed():
    a = parse(program(UNIFORM_BODY))
    b = parse(program(UNIFORM_BODY))
    c = parse(program("domain nx=9 ny=8\ninit uniform\nrun steps=1\n"))
    assert a.digest == b.digest
    assert a.digest != c.digest
    assert len(a.digest) == 64
    assert a.to_dict()["domain"] == {"nx": 8, "ny": 8}


def test_comments_and_blank_lines_are_ignored():
    text = "# leading comment\n\n" + SCHEME + "# body comment\n" + UNIFORM_BODY
    loose, tight = parse(text), parse(program(UNIFORM_BODY))
    loose_d, tight_d = loose.to_dict(), tight.to_dict()
    loose_d.pop("digest"), tight_d.pop("digest")
    assert loose_d == tight_d
    # the digest stays a digest of the exact submitted text (provenance)
    assert loose.digest != tight.digest


@pytest.mark.parametrize(
    "body, message",
    [
        ("domain nx=8 ny=8\ninit uniform\nrun steps=1\nwobble x=1\n", "unknown instruction"),
        ("domain nx=8 ny=9\ninit uniform\n", "missing instruction: run"),
        ("domain nx=8 ny=9\ninit uniform\nrun steps=1\npolynomial X&Y\n", "unsupported construct"),
        ("domain nx=8 ny=8\ninit uniform\nrun steps=1 extra=2\n", "unknown argument"),
        ("domain nx=8 ny=8\ninit uniform\nrun steps=1.5\n", "must be an integer"),
        ("domain nx=2 ny=8\ninit uniform\nrun steps=1\n", "at least 3x3"),
        (
            "domain nx=8 ny=8\nboundary ymin=free_slip\ninit uniform\nrun steps=1\n",
            "is not one of",
        ),
        ("domain nx=8 ny=8\ninit magic amp=1\nrun steps=1\n", "not implemented"),
    ],
)
def test_syntax_and_structure_errors(body, message):
    with pytest.raises(DSLError) as excinfo:
        parse(program(body))
    assert message in str(excinfo.value)


def test_expression_symbols_are_whitelisted():
    text = SCHEME.replace("polynomial X\n", "polynomial Z\n")
    with pytest.raises(DSLSyntaxError) as excinfo:
        parse(program(UNIFORM_BODY).replace(SCHEME, text))
    assert "unknown symbol 'Z'" in str(excinfo.value)


def test_expression_cannot_escape_the_sandbox():
    text = SCHEME.replace("equilibrium rho\n", "equilibrium __import__('os').getpid()\n")
    with pytest.raises(DSLSyntaxError):
        parse(program(UNIFORM_BODY).replace(SCHEME, text))


# ---------------------------------------------------------------------------
# semantic invariants of the lowered scheme
# ---------------------------------------------------------------------------
def test_d2q9_weights_and_moment_invariants():
    scheme = build_scheme(parse(program(UNIFORM_BODY)))
    expected = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
    assert np.allclose(scheme.weights, expected, atol=1e-14, rtol=0)
    assert scheme.checks["sum_w"] == pytest.approx(1.0, abs=1e-12)
    assert scheme.checks["sum_w_cx"] == pytest.approx(0.0, abs=1e-12)
    assert scheme.checks["sum_w_cxcy"] == pytest.approx(0.0, abs=1e-12)
    assert scheme.checks["sum_w_cx2"] == pytest.approx(1 / 3, abs=1e-12)
    assert scheme.checks["sum_w_cx2cy2"] == pytest.approx(1 / 9, abs=1e-12)
    assert scheme.checks["cs2_rel_error"] < 1e-12
    # nu derived from the declared shear relaxation s = 1.8
    assert scheme.nu == pytest.approx((1 / 1.8 - 0.5) / 3, rel=1e-14)
    # the declared equilibrium recovers mass and momentum at a generic state
    for key in ("feq_mass_rel_error", "feq_momentum_x_rel_error", "feq_momentum_y_rel_error"):
        assert scheme.checks[key] < 1e-12
    assert scheme.M.shape == (9, 9)


def test_taylor_green_field_is_divergence_free():
    ux, uy = taylor_green_velocity(32, 32, amp=0.05, mode=1)
    # centred differences on a periodic grid
    div = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / 2.0 + (
        np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
    ) / 2.0
    assert np.max(np.abs(div)) < 1e-14
    assert np.max(np.hypot(ux, uy)) == pytest.approx(0.05, rel=1e-12)


def test_equal_polynomials_are_rejected_as_singular():
    text = SCHEME.replace("polynomial Y\n", "polynomial X\n", 1)
    with pytest.raises(DSLSemanticError) as excinfo:
        build_scheme(parse(program(UNIFORM_BODY).replace(SCHEME, text)))
    assert "singular" in str(excinfo.value)


def test_singular_moment_matrix_detected():
    # replace the cross moment by the fourth-order one -> rank deficiency
    text = SCHEME.replace("polynomial X*Y\n", "polynomial X^2*Y^2\n", 1)
    with pytest.raises(DSLSemanticError) as excinfo:
        build_scheme(parse(program(UNIFORM_BODY).replace(SCHEME, text)))
    assert "singular" in str(excinfo.value)


def test_equilibrium_that_does_not_conserve_mass_is_rejected():
    text = SCHEME.replace("equilibrium rho\n", "equilibrium 2*rho\n", 1)
    prog = parse(program(UNIFORM_BODY).replace(SCHEME, text))
    with pytest.raises(DSLSemanticError) as excinfo:
        build_scheme(prog)
    assert "mass" in str(excinfo.value)


def test_relaxation_count_must_match_stencil():
    text = SCHEME.replace("relaxation 0,0,0,1.8,1.8,1.8,1.8,1.8,1.8\n", "relaxation 0,0,0,1.8\n")
    with pytest.raises(DSLSemanticError) as excinfo:
        parse(program(UNIFORM_BODY).replace(SCHEME, text))
    assert "relaxation has 4 rates" in str(excinfo.value)


@pytest.mark.parametrize("rate", ["2.0", "2.5", "0.0", "-0.5"])
def test_unstable_relaxation_is_rejected(rate):
    text = SCHEME.replace(
        "relaxation 0,0,0,1.8,1.8,1.8,1.8,1.8,1.8", f"relaxation 0,0,0,{rate},{rate},{rate},{rate},{rate},{rate}"
    )
    prog = parse(program(UNIFORM_BODY).replace(SCHEME, text))
    if float(rate) <= 0.0:
        with pytest.raises(DSLSemanticError) as excinfo:
            build_scheme(prog)
        assert "outside" in str(excinfo.value)
    else:
        with pytest.raises(DSLSemanticError) as excinfo:
            build_scheme(prog)
        assert "outside" in str(excinfo.value)


def test_unsupported_boundary_mix_is_refused():
    body = "domain nx=8 ny=8\nboundary xmin=bounce_back\ninit uniform\nrun steps=1\n"
    with pytest.raises(DSLSemanticError) as excinfo:
        run(program(body))
    assert "unsupported boundary mix" in str(excinfo.value)


def test_mach_guard_rejects_compressible_initial_state():
    body = "domain nx=8 ny=8\ninit uniform rho=1.0 ux=0.5\nrun steps=1\n"
    with pytest.raises(DSLSemanticError) as excinfo:
        run(program(body))
    assert "mach guard" in str(excinfo.value)


# ---------------------------------------------------------------------------
# lowering: the declared scheme must be the one that runs
# ---------------------------------------------------------------------------
def test_three_dimensional_stencil_is_refused_by_the_lowering():
    text = SCHEME.replace("dim=2", "dim=3").replace(
        "velocities=(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)",
        "velocities=(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)",
    )
    prog = parse(program(UNIFORM_BODY).replace(SCHEME, text))
    with pytest.raises(DSLSemanticError) as excinfo:
        build_scheme(prog)
    assert "dim=3" in str(excinfo.value)


def test_force_adds_exactly_one_momentum_unit_per_step():
    body = (
        "domain nx=4 ny=4\ninit uniform rho=1.0 ux=0.0 uy=0.0\n"
        "force gx=1e-6 gy=0.0\nrun steps=100\n"
    )
    result = run(program(body), sample_every=100)
    assert float(result.ux.mean()) == pytest.approx(100 * 1e-6, rel=1e-9)


def test_taylor_green_viscosity_matches_declared_relaxation():
    body = "domain nx=32 ny=32\ninit taylor_green rho=1.0 amp=0.05 mode=1\nrun steps=400\n"
    result = run(program(body), sample_every=2)
    k = 2.0 * np.pi / 32
    nu_eff, r2 = decay_rate_from_history(result.history, k)
    rel = abs(nu_eff - result.scheme.nu) / result.scheme.nu
    assert r2 > 0.999
    assert rel < 0.03  # strict band; measured 0.0001 in the xval run
    assert result.kinetic_energy < result.history[0]["kinetic_energy"]


def test_poiseuille_profile_matches_analytic_parabola():
    body = (
        "domain nx=8 ny=22\n"
        "boundary xmin=periodic xmax=periodic ymin=bounce_back ymax=bounce_back\n"
        "init uniform rho=1.0 ux=0.0 uy=0.0\n"
        "force gx=1e-5 gy=0.0\nrun steps=12000\n"
    )
    result = run(program(body), sample_every=500)
    err = poiseuille_error(result.ux, 22, nu=result.scheme.nu, body_force=1e-5)
    assert err["max_rel_error"] < 0.03  # measured 0.0069
    assert err["u_max_numeric"] > 0.0


# ---------------------------------------------------------------------------
# cross-validation against the OSS reference implementation (pylbm)
# ---------------------------------------------------------------------------
def test_scheme_matches_pylbm_reference():
    pytest.importorskip("sympy")
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stub = root / "tools" / "xval_pylbm_stub"
    cache = Path("/root/oss-cache/pylbm")
    if not cache.is_dir():
        pytest.skip("pylbm reference cache not available")
    sys.path.insert(0, str(stub))
    sys.path.insert(0, str(cache))
    try:
        from benchmarks.xval_instruction_dsl.run_xval import pylbm_reference  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import environment
        pytest.skip(f"pylbm reference not importable: {exc}")
    prog = parse(program(UNIFORM_BODY))
    ref = pylbm_reference(prog)
    assert ref["moment_matrix_max_abs_diff"] <= 1e-12
    assert ref["weights_max_abs_diff"] <= 1e-12
    assert ref["equilibrium_population_max_abs_diff"] <= 1e-12

# ---------------------------------------------------------------------------
# L3 integration: the capability is reachable from outside its own module
# ---------------------------------------------------------------------------
def test_cli_entry_roundtrip(tmp_path):
    """python -m tensorlbm.instruction_dsl_cli runs an instruction file end-to-end."""
    from tensorlbm.instruction_dsl_cli import run_program, run_program_file

    text = (
        "stencil dim=2 velocity=1.0 "
        "velocities=(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)\n"
        + "\n".join(f"polynomial {p}" for p in D2Q9_POLYNOMIALS) + "\n"
        + "\n".join(f"equilibrium {e}" for e in D2Q9_EQUILIBRIA) + "\n"
        "relaxation 0,0,0,1.8,1.8,1.8,1.8,1.8,1.8\n"
        "domain nx=16 ny=16\ninit taylor_green rho=1.0 amp=0.02 mode=1\n"
        "run steps=5\n"
    )
    rec = run_program(text)
    assert rec["steps"] == 5 and rec["domain"] == {"nx": 16, "ny": 16}
    assert rec["program_digest"]
    p = tmp_path / "tg.lbm"
    p.write_text(text)
    out = tmp_path / "run.json"
    rec2 = run_program_file(str(p), out=str(out))
    assert out.is_file() and rec2["program_digest"] == rec["program_digest"]


def test_example_harness_bridge_calls_dsl(tmp_path):
    """The main example runtime can drive a DSL program (real integration caller)."""
    from tensorlbm.example_harness import run_instruction_dsl_program

    text = (
        "stencil dim=2 velocity=1.0 "
        "velocities=(0,0);(1,0);(0,1);(-1,0);(0,-1);(1,1);(-1,1);(-1,-1);(1,-1)\n"
        + "\n".join(f"polynomial {p}" for p in D2Q9_POLYNOMIALS) + "\n"
        + "\n".join(f"equilibrium {e}" for e in D2Q9_EQUILIBRIA) + "\n"
        "relaxation 0,0,0,1.8,1.8,1.8,1.8,1.8,1.8\n"
        "domain nx=8 ny=8\ninit taylor_green rho=1.0 amp=0.02 mode=1\n"
        "run steps=3\n"
    )
    p = tmp_path / "tg.lbm"
    p.write_text(text)
    out = tmp_path / "bridge.json"
    rec = run_instruction_dsl_program(str(p), out=str(out))
    assert rec["steps"] == 3
    assert out.is_file()
    assert rec["program_digest"]
