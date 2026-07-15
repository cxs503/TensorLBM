"""Contract tests for domain-neutral collision and turbulence identities."""

from __future__ import annotations

from typing import cast

import pytest

from tensorlbm.core.collision import BGK, CUMULANT, MRT, CollisionModel
from tensorlbm.core.turbulence import NONE, SMAGORINSKY, WALE, TurbulenceModel


@pytest.mark.parametrize(
    "model, name",
    [(BGK, "BGK"), (MRT, "MRT"), (CUMULANT, "CUMULANT")],
)
def test_collision_identities_are_domain_neutral(model: CollisionModel, name: str) -> None:
    assert model.name == name
    assert isinstance(model, CollisionModel)


def test_collision_identities_only_describe_existing_model_families() -> None:
    assert {BGK.name, MRT.name, CUMULANT.name} == {"BGK", "MRT", "CUMULANT"}


@pytest.mark.parametrize(
    "model, name",
    [(NONE, "NONE"), (SMAGORINSKY, "SMAGORINSKY"), (WALE, "WALE")],
)
def test_turbulence_identities_are_domain_neutral(model: TurbulenceModel, name: str) -> None:
    assert model.name == name
    assert isinstance(model, TurbulenceModel)


def test_protocols_are_structural_for_adapters() -> None:
    class ExternalCollision:
        name = "EXTERNAL"

    class ExternalTurbulence:
        name = "EXTERNAL"

    assert isinstance(cast(CollisionModel, ExternalCollision()), CollisionModel)
    assert isinstance(cast(TurbulenceModel, ExternalTurbulence()), TurbulenceModel)
