"""Tests for the XVAL SOY free-energy drop-in core.

Anchor: ``XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION`` (TensorLBM vs
``ebinan92/Fingering_dynamics``, "cupy drop-in, no kernel fusion").

The tests assert *invariants* of the ported scheme rather than merely executing
code:

* the isotropic D2Q9 stencils are exact for quadratic fields (2nd-order
  consistency of ``grad`` and ``lap``), and the two backends share one
  implementation (identical array-op counts per step);
* the equilibrium populations carry the right moments
  (``sum f = rho``, ``sum g = psi``, ``sum f e + F/2 = rho u``);
* the order parameter is conserved by the periodic update;
* the physics check: a static droplet obeys the Young-Laplace law within the
  owner precision standard (15 %), with the *equilibrium* initial profile;
* true drop-in equivalence: NumPy and PyTorch backends agree to ~machine
  precision after hundreds of steps;
* the "no kernel fusion" premise is structurally true (no CUDA extension,
  no triton, no ``torch.compile`` in the implementation);
* the OSS reference adapter really executes OSS code (provenance hash) and
  agrees with the TensorLBM port when a checkout is present.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import numpy as np
import pytest

# --- worktree hygiene -------------------------------------------------------
# This developer box has an editable `tensorlbm` install pointing at *another*
# worktree; the tests must exercise the code of THIS checkout.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = _SRC.parent
import tensorlbm  # noqa: E402

if not str(pathlib.Path(tensorlbm.__file__).resolve()).startswith(str(_ROOT)):
    for _name in [k for k in list(sys.modules) if k == "tensorlbm" or k.startswith("tensorlbm.")]:
        del sys.modules[_name]
    importlib.invalidate_caches()
    import tensorlbm  # noqa: F811,E402

from tensorlbm.multiphase_soy_dropin import (  # noqa: E402
    E,
    SOYConfig,
    SOYFreeEnergyD2Q9,
    laplace_error_pct,
)

XI = 3.0
SIGMA = 0.01
TOL_PCT = 15.0


def _solver(backend: str = "numpy", **kw) -> SOYFreeEnergyD2Q9:
    cfg = SOYConfig(**{"nx": 32, "ny": 32, "radius": 8.0, "sigma": SIGMA,
                       "xi": XI, "backend": backend, **kw})
    s = SOYFreeEnergyD2Q9(cfg)
    s.init_droplet()
    return s


# ---------------------------------------------------------------------------
# stencils / moments (cheap, no time stepping)
# ---------------------------------------------------------------------------
def test_isotropic_gradient_is_exact_for_quadratic_field():
    s = _solver()
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    f = xx.astype(float) ** 2 + 3.0 * xx * yy
    gx = s.b.to_numpy(s._grad(s.b.from_numpy(f), 1))[1:-1, 1:-1]
    gy = s.b.to_numpy(s._grad(s.b.from_numpy(f), 0))[1:-1, 1:-1]
    # interior only: a global quadratic is not periodic, the stencil is exact
    # wherever both stencil rows/columns are fully inside the box.
    assert np.allclose(gx, (2.0 * xx + 3.0 * yy)[1:-1, 1:-1], atol=1e-10)
    assert np.allclose(gy, (3.0 * xx)[1:-1, 1:-1], atol=1e-10)


def test_isotropic_laplacian_is_exact_for_quadratic_field():
    s = _solver()
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    q = (xx.astype(float) ** 2 + yy.astype(float) ** 2)
    lap = s.b.to_numpy(s._laplacian(s.b.from_numpy(q)))[1:-1, 1:-1]
    assert np.allclose(lap, 4.0, atol=1e-10)


def test_equilibrium_populations_carry_the_expected_moments():
    s = _solver()
    rho = s.b.to_numpy(s.rho)
    psi = s.b.to_numpy(s.psi)
    fsum = s.b.to_numpy(s.b.sum_pop([s.f[i] for i in range(9)]))
    gsum = s.b.to_numpy(s.b.sum_pop([s.g[i] for i in range(9)]))
    assert np.allclose(fsum, rho, atol=1e-12)
    assert np.allclose(gsum, psi, atol=1e-12)


def test_momentum_definition_matches_half_force_correction():
    s = _solver()
    s.run(5)
    jx = s.b.to_numpy(s.b.sum_pop([s.f[i] * float(E[i][0]) for i in range(9)]))
    fx = s.b.to_numpy(s.mu * s.grad_x)
    rho = s.b.to_numpy(s.rho)
    ux = s.b.to_numpy(s.ux)
    assert np.allclose(jx + 0.5 * fx, rho * ux, atol=1e-10)


def test_mobility_is_unity_when_viscosity_ratio_is_one():
    s = _solver()
    tau = s.b.to_numpy(s.mix_tau)
    assert np.allclose(tau, s.tau_f, atol=1e-6)


# ---------------------------------------------------------------------------
# the two backends are one implementation ("drop-in")
# ---------------------------------------------------------------------------
def test_backends_report_identical_array_op_counts():
    a, b = _solver("numpy"), _solver("torch")
    a.run(20)
    b.run(20)
    assert a.ops_per_step() == b.ops_per_step()
    assert a.ops_per_step() > 0


def test_backends_agree_to_machine_precision_after_200_steps():
    a, b = _solver("numpy"), _solver("torch")
    a.run(200)
    b.run(200)
    ma, mb = a.measure(), b.measure()
    assert abs(ma["dp"] - mb["dp"]) / abs(ma["dp"]) < 1e-10
    assert abs(ma["u_max"] - mb["u_max"]) <= 1e-12 + 1e-6 * ma["u_max"]
    psi_a = a.b.to_numpy(a.psi)
    psi_b = b.b.to_numpy(b.psi)
    assert np.max(np.abs(psi_a - psi_b)) < 1e-10


# ---------------------------------------------------------------------------
# physics: conservation + Young-Laplace
# ---------------------------------------------------------------------------
def test_order_parameter_is_conserved_by_the_periodic_update():
    """sum(psi) is the conserved quantity; |psi| is not (the interface relaxes)."""
    s = _solver()
    before = s.measure()["psi_sum"]
    s.run(100)
    after = s.measure()["psi_sum"]
    assert abs(after - before) / max(abs(before), 1.0) < 1e-8


def test_static_droplet_obeys_young_laplace_law_within_15_percent():
    cfg = SOYConfig(nx=60, ny=60, radius=15.0, sigma=SIGMA, xi=XI, backend="numpy")
    s = SOYFreeEnergyD2Q9(cfg)
    s.init_droplet()
    s.run(1000)
    m = s.measure()
    err = laplace_error_pct(m["dp"], SIGMA, 15.0)
    assert abs(err) < TOL_PCT, f"Laplace error {err:+.3f}% exceeds {TOL_PCT}%"
    # residual flow must be tiny for a *static* droplet
    assert m["u_max"] < 1e-4
    # independent surface-tension route (capillary energy) must be consistent
    assert abs(m["sigma_rel_err_pct"]) < TOL_PCT


def test_sharp_initial_profile_is_a_slow_transient_not_an_equilibrium():
    """The initial interface width dominates the transient: documented failure mode."""
    eq = SOYConfig(nx=60, ny=60, radius=15.0, backend="numpy", init_profile="equilibrium")
    sharp = SOYConfig(nx=60, ny=60, radius=15.0, backend="numpy", init_profile="sharp")
    a, b = SOYFreeEnergyD2Q9(eq), SOYFreeEnergyD2Q9(sharp)
    a.init_droplet()
    b.init_droplet()
    a.run(600)
    b.run(600)
    err_eq = laplace_error_pct(a.measure()["dp"], SIGMA, 15.0)
    err_sharp = laplace_error_pct(b.measure()["dp"], SIGMA, 15.0)
    assert abs(err_eq) < TOL_PCT
    assert abs(err_sharp) > abs(err_eq)          # the sharp start is worse
    assert err_sharp > 0                          # and biased high


def test_pressure_form_switch_changes_the_result_but_stays_acceptable():
    results = {}
    for form in ("oss", "psi_mu_over_3"):
        cfg = SOYConfig(nx=48, ny=48, radius=12.0, backend="numpy", pressure_form=form)
        s = SOYFreeEnergyD2Q9(cfg)
        s.init_droplet()
        s.run(800)
        results[form] = laplace_error_pct(s.measure()["dp"], SIGMA, 12.0)
    for form, err in results.items():
        assert abs(err) < TOL_PCT, f"{form}: {err:+.3f}%"
    assert abs(results["oss"] - results["psi_mu_over_3"]) > 1e-3


# ---------------------------------------------------------------------------
# "no kernel fusion" premise + solver-layer integration
# ---------------------------------------------------------------------------
def test_implementation_contains_no_fused_or_custom_kernels():
    src = pathlib.Path(
        importlib.import_module("tensorlbm.multiphase_soy_dropin").__file__
    ).read_text(encoding="utf-8")
    # strip docstrings: the module *documents* that it avoids these mechanisms
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body.pop(0)
                if not body:
                    body.append(ast.Pass())
    code = ast.unparse(tree)
    banned = ["load_inline", "cpp_extension", "triton", "num_warps", "torch.compile",
              "RawKernel", "extern \"C\""]
    hits = [token for token in banned if token in code]
    assert hits == [], f"custom/fused kernel machinery found: {hits}"


def test_solver_layer_caller_returns_laplace_diagnostics():
    from tensorlbm.multiphase_static_droplet import soy_free_energy_droplet

    out = soy_free_energy_droplet(radius=8.0, nx=32, ny=32, steps=50, backend="numpy")
    assert set({"dp", "err_pct", "u_max", "config"}) <= set(out)
    assert out["config"]["backend"] == "numpy"
    assert np.isfinite(out["dp"])


# ---------------------------------------------------------------------------
# external reference arm
# ---------------------------------------------------------------------------
def _oss_adapter():
    path = pathlib.Path(__file__).resolve().parents[1] / "xval" / "cupy_dropin_no_kernel_fusion"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("oss_adapter")


def test_oss_reference_provenance_and_agreement():
    mod = _oss_adapter()
    ref = mod.OSSReference(nx=32, ny=32, radius=8.0, sigma=SIGMA, xi=XI)
    prov = ref.provenance()
    if not prov["available"]:
        pytest.skip(f"OSS checkout unavailable: {prov['reason']}")
    assert len(prov["oss_sha256"]) == 64
    ref.setup()
    out = ref.run_and_measure(200)
    # same physics, same transient: the two implementations must track each other
    s = SOYFreeEnergyD2Q9(SOYConfig(nx=32, ny=32, radius=8.0, sigma=SIGMA, xi=XI,
                                    backend="numpy"))
    s.init_droplet()
    s.run(200)
    mine = s.measure()["dp"]
    assert abs(out["dp"] - mine) / abs(mine) < 0.02
    assert np.isfinite(out["dp"])