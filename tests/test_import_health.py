"""Import-health gate for the ``tensorlbm`` package.

Reusable verification surface: every module in :mod:`tensorlbm` must at least
import.  A module that cannot be imported is invisible to every other test
(collection error) and to users — this gate turns that silent rot into a red
test with the failing symbol named.

Background (verification debt closed by m14806): a sweep of all 653 modules of
the package found 4 import-time defects that no test could see because the
only tests covering those modules could not even be collected:

* ``cad`` — re-exported ``optimal_advance_ratio`` / ``propeller_design`` /
  ``propeller_disk_mask`` / ``wageningen_b_series`` from ``propeller_cad``,
  none of which exist there any more;
* ``boussinesq`` and ``scalar_verification_in65`` — imported the pre-TRT-rewrite
  D2Q5 names (``C5``/``W5``/``_get_w5``/``scalar_collision``/
  ``scalar_equilibrium``/``scalar_macroscopic``/``scalar_stream``/
  ``scalar_diffusivity``) that ``scalar_transport`` no longer declares;
* ``drag_monitor`` — docstring claimed a ``DragMonitor`` re-export, file had
  no import statement at all.

The repair (m14806) restored the legacy names as explicit adapters in
``scalar_transport``, pruned the stale ``cad`` re-exports, and gave
``drag_monitor`` a real re-export.  This test is the regression gate.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import tensorlbm


MODULES = sorted(m.name for m in pkgutil.iter_modules(tensorlbm.__path__))


@pytest.mark.unit
def test_package_has_modules():
    assert len(MODULES) > 500, f"only {len(MODULES)} modules discovered"


@pytest.mark.unit
@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"tensorlbm.{name}")


# ---------------------------------------------------------------------------
# The four defects that motivated the gate — pinned explicitly so a regression
# names the user-facing import path, not just a module name.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_drag_monitor_reexport():
    from tensorlbm.drag_monitor import DragMonitor
    from tensorlbm.yplus_guide import DragMonitor as Canonical

    assert DragMonitor is Canonical


@pytest.mark.unit
def test_cad_namespace_imports():
    from tensorlbm import cad

    from tensorlbm.propeller_cad import build_propeller_mask  # noqa: F401

    for stale in ("optimal_advance_ratio", "propeller_design", "propeller_disk_mask"):
        assert not hasattr(cad, stale), (
            f"cad still advertises {stale}, which propeller_cad does not define"
        )


@pytest.mark.unit
def test_legacy_d2q5_names_are_adapters():
    """The legacy names must be thin adapters, not a second implementation."""
    import torch

    from tensorlbm import scalar_transport as st

    assert st.C5.shape == (5, 2)
    assert abs(float(st.W5.sum()) - 1.0) < 1e-6
    assert torch.equal(st.W5, st._get_w5(torch.float32))

    c = torch.full((6, 5), 0.3)
    ux = torch.zeros(6, 5)
    uy = torch.zeros(6, 5)
    assert torch.equal(st.scalar_equilibrium(c, ux, uy), st.equilibrium_scalar_d2q5(c, ux, uy))
    g = st.scalar_equilibrium(c, ux, uy)
    assert torch.equal(st.scalar_macroscopic(g), st.scalar_field(g))
    # single-relaxation legacy collision == BGK degenerate TRT
    assert torch.equal(
        st.scalar_collision(g, c, ux, uy, 0.7),
        st.collide_scalar_trt(g, c, ux, uy, 0.7, None),
    )
    # legacy streaming drops the mass-budget tuple
    streamed = st.scalar_stream(g)
    assert isinstance(streamed, torch.Tensor) and streamed.shape == g.shape
    assert st.scalar_diffusivity(1.5) == pytest.approx(st.tau_from_diffusivity(1.0) * 3.0 / 3.0 if False else st.CS2_D2Q5 * 1.0)