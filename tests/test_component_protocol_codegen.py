"""Invariant tests for the component protocol + template codegen.

These assert *properties*, not just that the code runs:

* lattice invariants of ``D2Q9`` hold and a corrupted stencil is rejected;
* the component protocol rejects inconsistent declarations (a component whose
  claimed viscosity does not follow from the relaxation it emits);
* template rendering is strict (missing/extra/residual placeholders raise);
* generation is deterministic (sha256) and the emitted source is a specialised
  kernel (constants baked as literals, no residual field);
* the generated kernel equals an independently written reference kernel to
  machine precision, for both collision families;
* the generated kernel reproduces the analytic Taylor-Green decay rate, i.e.
  the viscosity the ``ComponentSpec`` declares (15% owner criterion, asserted
  at a stricter 5% to keep margin).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "xval" / "component_protocol_codegen"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.codegen import (  # noqa: E402
    BGKComponent,
    D2Q9,
    ComponentSpec,
    GeneratedKernel,
    ProtocolViolation,
    StencilSpec,
    StencilViolation,
    TRTComponent,
    available_components,
    generate_kernel,
    get_component,
    reference_collide,
    reference_moments,
    reference_step,
    validate_component,
)
from tensorlbm.codegen.templates import (  # noqa: E402
    MODULE_TEMPLATE,
    render,
    template_fields,
)

Q = D2Q9.q


def _random_field(seed: int = 0, shape=(Q, 10, 8)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1.0 / Q + 0.01 * rng.standard_normal(shape)


# ------------------------------------------------------------------ stencil
def test_d2q9_stencil_invariants_hold():
    assert D2Q9.violations() == []


def test_corrupted_stencil_is_rejected():
    broken = StencilSpec(
        name="BAD",
        ex=D2Q9.ex,
        ey=D2Q9.ey,
        w=(4 / 9,) + (1 / 9,) * 4 + (1 / 36,) * 4,
        opposite=(0, 3, 4, 1, 2, 7, 8, 6, 5),  # (5,7)/(6,8) pairs broken
    )
    bad = broken.violations()
    assert any("involutive" in b or "is not -e" in b for b in bad)
    with pytest.raises(StencilViolation):
        broken.validate()


# ----------------------------------------------------------------- protocol
@pytest.mark.parametrize("component", [BGKComponent(tau=0.56), TRTComponent(tau=0.56)])
def test_protocol_accepts_shipped_components(component):
    report = validate_component(component)
    assert report["stencil"] == "D2Q9"
    assert report["family"] in {"BGK", "TRT"}
    assert abs(report["transport"]["nu"] - 0.02) < 1e-12


def test_protocol_rejects_declared_viscosity_mismatch():
    class Lying(BGKComponent):
        registry_key = "LYING_TEST"

        def transport_coefficients(self):
            return {"nu": 0.5}  # not what 1/tau implies

    with pytest.raises(ProtocolViolation) as exc:
        validate_component(Lying(tau=0.56))
    assert "declared nu" in str(exc.value)


def test_protocol_rejects_inconsistent_magic():
    class BadMagic(TRTComponent):
        registry_key = "BADMAGIC_TEST"

        def s_minus(self) -> float:
            return 1.0  # inconsistent with magic = 3/16

    with pytest.raises(ProtocolViolation) as exc:
        validate_component(BadMagic(tau=0.56, magic=3.0 / 16.0))
    assert "magic" in str(exc.value)


def test_protocol_rejects_missing_emission_and_bad_tau():
    class Silent(BGKComponent):
        registry_key = "SILENT_TEST"

        def emit_collision(self, stencil):
            return []

    with pytest.raises(ProtocolViolation) as exc:
        validate_component(Silent(tau=0.56))
    assert "no statements" in str(exc.value)

    with pytest.raises(ProtocolViolation):
        BGKComponent(tau=0.5)  # zero viscosity is outside the protocol

    # ``make`` is a plain constructor; ``validate`` is where the protocol bites.
    with pytest.raises(ProtocolViolation):
        ComponentSpec.make("x", "MRT", tau=0.6).validate()  # family not emitted here
    with pytest.raises(ProtocolViolation):
        ComponentSpec.make("x", "TRT", tau=0.6).validate()  # TRT requires 'magic'



def test_registry_shape():
    assert {"BGK", "TRT"} <= set(available_components())
    assert get_component("BGK") is BGKComponent
    with pytest.raises(KeyError):
        get_component("NOPE")


# ---------------------------------------------------------------- templates
def test_template_rendering_is_strict():
    fields = template_fields(MODULE_TEMPLATE)
    assert {"collision_block", "stream_block", "moments_block"} <= fields

    with pytest.raises(KeyError):
        render(MODULE_TEMPLATE, {})

    subs = {name: "    pass" for name in fields}
    subs.update({"q": 9, "ex": (0,), "ey": (0,), "w": "(1.0,)", "name": "x"})
    with pytest.raises(ValueError):
        render("plain {a}", {"a": 1, "b": 2})
    with pytest.raises(ValueError):
        render("left {a} over", {"a": "{b}"})


# ---------------------------------------------------------------- generator
@pytest.mark.parametrize("build", [lambda: BGKComponent(tau=0.56), lambda: TRTComponent(tau=0.56)])
def test_generation_is_deterministic_and_specialised(build):
    k1, k2 = generate_kernel(build()), generate_kernel(build())
    assert isinstance(k1, GeneratedKernel)
    assert k1.sha256 == k2.sha256
    # constants are baked, not looked up at run time
    assert "S_PLUS = " in k1.source
    assert repr(1.0 / 0.56) in k1.source
    # every direction is unrolled explicitly
    for i in range(Q):
        assert f"p{i} =" in k1.source
    assert "{" not in k1.source
    k3 = generate_kernel(BGKComponent(tau=0.7))
    assert k3.sha256 != generate_kernel(BGKComponent(tau=0.56)).sha256


def test_equilibrium_matches_reference():
    from tensorlbm.codegen import reference_equilibrium

    k = generate_kernel(BGKComponent(tau=0.56))
    rng = np.random.default_rng(3)
    rho = 1.0 + 0.01 * rng.standard_normal((8, 6))
    ux = 0.02 * rng.standard_normal((8, 6))
    uy = 0.02 * rng.standard_normal((8, 6))
    assert np.abs(k.equilibrium(rho, ux, uy) - reference_equilibrium(rho, ux, uy)).max() < 1e-15


@pytest.mark.parametrize(
    "build,family", [(lambda: BGKComponent(tau=0.56), "BGK"), (lambda: TRTComponent(tau=0.56), "TRT")]
)
def test_generated_kernel_equals_independent_reference(build, family):
    k = generate_kernel(build())
    f = _random_field()
    assert np.abs(k.collide(f) - reference_collide(f, family=family, tau=0.56)).max() < 1e-12
    assert np.abs(k.step(f) - reference_step(f, family=family, tau=0.56)).max() < 1e-12
    fg, fr = f.copy(), f.copy()
    for _ in range(20):
        fg, fr = k.step(fg), reference_step(fr, family=family, tau=0.56)
    assert np.abs(fg - fr).max() < 1e-12


def test_collide_conserves_mass_and_momentum():
    for build in (lambda: BGKComponent(tau=0.56), lambda: TRTComponent(tau=0.56)):
        k = generate_kernel(build())
        f = _random_field(seed=7)
        post = k.collide(f)
        assert np.abs(post.sum(0) - f.sum(0)).max() < 1e-15
        for ex, ey in ((D2Q9.ex, "x"), (D2Q9.ey, "y")):
            before = np.tensordot(np.asarray(ex, float), f, axes=(0, 0))
            after = np.tensordot(np.asarray(ex, float), post, axes=(0, 0))
            assert np.abs(after - before).max() < 1e-15, ey


# ------------------------------------------------------------------- physics
@pytest.mark.parametrize("family", ["BGK", "TRT"])
def test_taylor_green_decay_matches_analytic_viscosity(family):
    """Generated kernel reproduces the declared viscosity on the analytic case."""
    import run_arm  # xval harness, reused so the test and the artifact agree

    build = (lambda: BGKComponent(tau=0.56)) if family == "BGK" else (lambda: TRTComponent(tau=0.56))
    component = build()
    kernel = generate_kernel(component)
    n, u0 = 32, 0.02
    k_lb = run_arm.lattice_wavenumber(n)

    ux0, uy0 = run_arm.taylor_green_fields(n, u0)
    f = kernel.equilibrium(np.ones_like(ux0), ux0, uy0)
    samples = []
    for step in range(121):
        if step % 10 == 0:
            _rho, ux, uy = kernel.moments(f)
            samples.append({"t": step, "E": run_arm.kinetic_energy(ux, uy)})
        f = kernel.step(f)
    nu_meas = run_arm.fit_viscosity(samples, 20, k_lb)["nu_measured"]
    nu_decl = component.transport_coefficients()["nu"]
    err = abs(nu_meas - nu_decl) / nu_decl
    assert err < 0.05, f"{family}: nu_measured={nu_meas} vs declared={nu_decl} ({err:.2%})"


def test_reference_moments_are_consistent_with_equilibrium():
    """At equilibrium the reference moments return the fields they were built from."""
    from tensorlbm.codegen import reference_equilibrium

    rng = np.random.default_rng(11)
    rho = 1.0 + 0.02 * rng.standard_normal((6, 5))
    ux = 0.03 * rng.standard_normal((6, 5))
    uy = 0.03 * rng.standard_normal((6, 5))
    feq = reference_equilibrium(rho, ux, uy)
    r, vx, vy = reference_moments(feq)
    assert np.abs(r - rho).max() < 1e-15
    assert np.abs(vx - ux).max() < 1e-15
    assert np.abs(vy - uy).max() < 1e-15