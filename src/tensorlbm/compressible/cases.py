"""Verification case configurations for compressible shock wave simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalShockConfig:
    """Configuration for normal shock verification."""
    M1: float
    gamma: float = 1.4
    nx: int = 256
    n_steps: int = 500
    shock_pos: float = 0.5

    @property
    def label(self) -> str:
        return f"normal_shock_M{self.M1:.0f}_nx{self.nx}"


@dataclass
class DoubleMachReflectionConfig:
    """Configuration for double Mach reflection (Woodward-Colella)."""
    M1: float = 10.0
    theta_deg: float = 30.0
    gamma: float = 1.4
    nx: int = 512
    ny: int = 256

    @property
    def label(self) -> str:
        return f"dmr_M{self.M1:.0f}_theta{self.theta_deg:.0f}"


@dataclass
class ForwardStepConfig:
    """Configuration for forward-facing step."""
    M1: float = 3.0
    step_height_ratio: float = 0.2
    gamma: float = 1.4
    nx: int = 512
    ny: int = 128

    @property
    def label(self) -> str:
        return f"forward_step_M{self.M1:.0f}_h{self.step_height_ratio:.1f}"


# Pre-defined parameter scan configurations
NORMAL_SHOCK_SCAN = [
    NormalShockConfig(M1=2.0, nx=128),
    NormalShockConfig(M1=2.0, nx=256),
    NormalShockConfig(M1=2.0, nx=512),
    NormalShockConfig(M1=3.0, nx=128),
    NormalShockConfig(M1=3.0, nx=256),
    NormalShockConfig(M1=3.0, nx=512),
    NormalShockConfig(M1=5.0, nx=128),
    NormalShockConfig(M1=5.0, nx=256),
    NormalShockConfig(M1=5.0, nx=512),
    NormalShockConfig(M1=8.0, nx=128),
    NormalShockConfig(M1=8.0, nx=256),
    NormalShockConfig(M1=8.0, nx=512),
    NormalShockConfig(M1=10.0, nx=128),
    NormalShockConfig(M1=10.0, nx=256),
    NormalShockConfig(M1=10.0, nx=512),
]
