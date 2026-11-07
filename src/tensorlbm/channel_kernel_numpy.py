"""Lean D3Q19 channel kernel (NumPy) for wall-resolved turbulence campaigns.

Why this module exists
----------------------
The production 3-D solver lives in :mod:`tensorlbm.boundary.wall_resolved_channel3d`
and is written as a sequence of ``torch`` element-wise kernels.  Its per-step
cost is dominated by kernel-launch/thread-synchronisation overhead rather than
by floating-point work: measured on a 128-core node it delivers ~0.5-0.8
Mcell-step/s almost independently of the grid size, so a Re_tau = 180 DNS that
needs ~10^10 cell-steps costs days.  That, and not the physics, is what blocked
the earlier turbulent-channel node (see ``docs/reports/P20-N01-...``).

This module implements the *same* discrete scheme in NumPy with one allocation
per operator instead of ~180 tiny kernels:

* D3Q19, second-order equilibrium, ``BGK`` **or** two-relaxation-time (TRT,
  magic parameter ``Lambda = 3/16``) collision, selected per call;
* body force through the *split* Guo correction used by the production wall
  kernel (force injected post-collision, no half-step factor), so the steady
  state satisfies ``tau_w = rho a h / 2`` exactly;
* periodic ``x``/``z``, two solid wall rows in ``y`` with the node-based
  halfway bounce-back of the production code (``f[i] <- f[opp(i)]`` at wall
  nodes after streaming), giving the same effective wall location
  ``y_w ~= 0.64`` cell;
* mass / momentum diagnostics for fail-closed checks.

The array layout is ``(19, nx, ny, nz)`` to match the production solver.

Validation
----------
``tests/test_d3q19_channel_numpy.py`` pins

1. **bit-level agreement with the torch pipeline** - one step of this kernel
   (BGK) equals ``collide_bgk3d -> where(mask) -> Guo -> stream3d ->
   bounce_back_cells_3d`` to ~1e-15, which proves streaming, bounce-back,
   force and diagnostics are the *same* discrete operators;
2. **analytic Poiseuille recovery** - laminar body-force channel converges to
   ``u(y) = (a/(2 nu)) (y-y_w)(H-(y-y_w))`` with the measured wall origin;
3. **negative control** - with no perturbation the solver stays laminar (zero
   Reynolds shear stress), i.e. the turbulent state is *not* hard-coded.

Only NumPy is required; the module is deliberately dependency-light so it can
run on the loong64 cluster nodes without a CUDA/torch stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# D3Q19 lattice (same ordering as :mod:`tensorlbm.d3q19`).
C = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=np.int64,
)

W = np.array(
    [
        1 / 3,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 18,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
        1 / 36,
    ],
    dtype=np.float64,
)

OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])

CS2 = 1.0 / 3.0

__all__ = [
    "ChannelKernelConfig",
    "ChannelKernelState",
    "equilibrium",
    "macroscopic",
    "collide",
    "stream",
    "bounce_back_walls",
    "apply_body_force",
    "step",
    "run_channel",
    "poiseuille_exact",
]


@dataclass(frozen=True)
class ChannelKernelConfig:
    """Grid, physics and sampling controls for :func:`run_channel`."""

    nx: int = 128
    ny: int = 48
    nz: int = 64
    re_tau: float = 180.0
    u_tau: float = 0.006
    steps: int = 20000
    warmup_steps: int = 3000
    sample_interval: int = 5
    collision: str = "trt"
    magic_lambda: float = 3.0 / 16.0
    initialization: str = "spectral"
    perturbation: float = 0.05
    spectral_modes: int = 24
    max_wavenumber: int = 3
    seed: int = 20261106
    dtype: str = "float64"

    @property
    def height(self) -> int:
        """Number of fluid rows between the two solid wall rows."""
        return self.ny - 2

    @property
    def nu(self) -> float:
        return self.u_tau * (0.5 * self.height) / self.re_tau

    @property
    def tau(self) -> float:
        return 0.5 + 3.0 * self.nu

    @property
    def body_force_acceleration(self) -> float:
        return 2.0 * self.u_tau**2 / self.height

    @property
    def dx_plus(self) -> float:
        return self.u_tau / self.nu


@dataclass
class ChannelKernelState:
    f: np.ndarray
    config: ChannelKernelConfig
    solid: np.ndarray = field(repr=False)
    step: int = 0


def _shift(f: np.ndarray, direction: int) -> np.ndarray:
    """Stream one population (periodic in all three axes)."""
    cx, cy, cz = C[direction]
    if cx == 0 and cy == 0 and cz == 0:
        return f
    return np.roll(f, (cx, cy, cz), axis=(0, 1, 2))


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """Second-order D3Q19 equilibrium, shape ``(19, nx, ny, nz)``."""
    out = np.empty((19,) + rho.shape, dtype=rho.dtype)
    usq = ux * ux + uy * uy + uz * uz
    for i in range(19):
        cu = C[i, 0] * ux + C[i, 1] * uy + C[i, 2] * uz
        out[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
    return out


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``rho, ux, uy, uz`` from the first two moments of ``f``."""
    rho = f.sum(axis=0)
    inv = 1.0 / rho
    ux = (C[:, 0, None, None, None] * f).sum(axis=0) * inv
    uy = (C[:, 1, None, None, None] * f).sum(axis=0) * inv
    uz = (C[:, 2, None, None, None] * f).sum(axis=0) * inv
    return rho, ux, uy, uz


def collide(
    f: np.ndarray,
    tau: float,
    *,
    collision: str = "trt",
    magic_lambda: float = 3.0 / 16.0,
) -> np.ndarray:
    """Collision step (BGK or TRT) about the local equilibrium."""
    rho, ux, uy, uz = macroscopic(f)
    feq = equilibrium(rho, ux, uy, uz)
    s_plus = 1.0 / tau
    if collision == "bgk":
        return f - s_plus * (f - feq)
    if collision != "trt":
        raise ValueError(f"unknown collision model {collision!r}")
    # Two-relaxation-time: split into symmetric / antisymmetric parts.
    s_minus = 1.0 / (0.5 + magic_lambda / (tau - 0.5))
    f_plus = 0.5 * (f + f[OPP])
    f_minus = 0.5 * (f - f[OPP])
    eq_plus = 0.5 * (feq + feq[OPP])
    eq_minus = 0.5 * (feq - feq[OPP])
    return f - s_plus * (f_plus - eq_plus) - s_minus * (f_minus - eq_minus)


def apply_body_force(
    f: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    fx: np.ndarray,
) -> np.ndarray:
    """Guo split forcing for a purely streamwise body force.

    Mirrors :func:`tensorlbm.boundary.wall_model.guo_body_force_d3q19`: the
    force is added post-collision using the post-collisionvelocity field, so
    the discrete momentum balance ``rho a h / 2 = tau_w`` holds in the steady
    state without a half-step factor.
    """
    out = f.copy()
    uf = ux * fx
    for i in range(19):
        cx, cy, cz = C[i]
        c_dot_f = cx * fx
        out[i] += W[i] * (3.0 * (c_dot_f - uf) + 9.0 * (cx * ux + cy * uy + cz * uz) * c_dot_f)
    return out


def stream(f: np.ndarray) -> np.ndarray:
    """Full periodic streaming (wall rows overwritten by bounce-back)."""
    out = np.empty_like(f)
    out[0] = f[0]
    for i in range(1, 19):
        out[i] = _shift(f[i], i)
    return out


def bounce_back_walls(f: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Node-based halfway bounce-back on the solid wall rows.

    ``f[i] <- f[opp(i)]`` at every solid node, i.e. the post-streaming
    populations are reflected back into the fluid.  This is exactly the
    operator used by the production solver
    (:func:`tensorlbm.boundary.boundaries3d.bounce_back_cells_3d`).
    """
    out = f.copy()
    mask = solid[None]
    out[:] = np.where(mask, f[OPP], f)
    return out


def step(state: ChannelKernelState) -> None:
    """Advance the channel by one lattice step (in place on ``state.f``)."""
    cfg = state.config
    f = state.f
    solid = state.solid
    solid_3d = solid[None]
    collided = collide(f, cfg.tau, collision=cfg.collision, magic_lambda=cfg.magic_lambda)
    post = np.where(solid_3d, f, collided)
    rho, ux, uy, uz = macroscopic(post)
    fx = np.where(solid, 0.0, rho * cfg.body_force_acceleration)
    post = apply_body_force(post, ux, uy, uz, fx)
    streamed = stream(post)
    state.f = bounce_back_walls(streamed, solid)
    state.step += 1


def poiseuille_exact(
    ny: int,
    nu: float,
    body_force: float,
    wall_origin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic plane-Poiseuille profile of the discrete channel.

    Returns ``(y_centre, u(y))`` on the fluid cell centres, with the effective
    wall at ``y_w = wall_origin`` and the free surface condition
    ``u = 0`` at both walls.
    """
    height = ny - 2
    y = np.arange(1, ny - 1, dtype=float)
    s = y - wall_origin
    upper = (height + 1.0) - wall_origin
    u = body_force / (2.0 * nu) * s * (upper - s)
    return y, u


def _spectral_perturbation(
    cfg: ChannelKernelConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divergence-free random perturbation, tapered to zero at both walls.

    White noise is projected onto the plane normal to ``k`` in Fourier space
    (exact discrete solenoidality), transformed back, then multiplied by a
    ``sin(pi (y+1/2)/H)`` envelope so the perturbation vanishes on the walls.
    The envelope re-introduces an O(1/H) compressive residue near the wall,
    which is harmless: it only seeds acoustic waves that the LBM propagates
    away during the warm-up.
    """
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    raw = rng.standard_normal((3, nx, ny, nz))
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kz = np.fft.fftfreq(nz) * nz
    kkx, kky, kkz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kkx**2 + kky**2 + kkz**2
    k2[0, 0, 0] = 1.0
    spectrum = np.fft.fftn(raw, axes=(1, 2, 3))
    dot = kkx * spectrum[0] + kky * spectrum[1] + kkz * spectrum[2]
    for comp, kk in enumerate((kkx, kky, kkz)):
        spectrum[comp] -= kk * dot / k2
    spectrum[..., 0, 0, 0] = 0.0
    field = np.real(np.fft.ifftn(spectrum, axes=(1, 2, 3)))
    height = ny - 2
    envelope = np.sin(np.pi * (np.arange(ny) - 0.5) / (height + 1.0))[None, :, None]
    envelope = np.clip(envelope, 0.0, None)
    field *= envelope[None]
    field[:, :, 0, :] = 0.0
    field[:, :, -1, :] = 0.0
    return field[0], field[1], field[2]


def make_state(cfg: ChannelKernelConfig) -> ChannelKernelState:
    """Initialise a channel state with the log-law mean plus a perturbation."""
    dtype = np.dtype(cfg.dtype)
    solid = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=bool)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    y = np.arange(1, cfg.ny - 1, dtype=float)
    y_plus = (y - 0.5) * cfg.dx_plus
    kappa, b_log = 0.41, 5.0
    u_plus = np.where(
        y_plus < 11.0,
        y_plus,
        np.log(np.maximum(y_plus, 1.0)) / kappa + b_log,
    )
    mean = np.zeros((cfg.ny,))
    mean[1:-1] = u_plus * cfg.u_tau
    ux = np.broadcast_to(mean[None, :, None], (cfg.nx, cfg.ny, cfg.nz)).astype(dtype).copy()
    uy = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=dtype)
    uz = np.zeros((cfg.nx, cfg.ny, cfg.nz), dtype=dtype)
    if cfg.initialization == "spectral":
        rng = np.random.default_rng(cfg.seed)
        px, py, pz = _spectral_perturbation(cfg, rng)
        norm = np.sqrt(np.mean(px**2 + py**2 + pz**2))
        scale = cfg.perturbation * cfg.u_tau / max(norm, 1e-30)
        ux = ux + scale * px
        uy = uy + scale * py
        uz = uz + scale * pz
    elif cfg.initialization != "laminar":
        raise ValueError(f"unknown initialization {cfg.initialization!r}")
    ux[solid] = 0.0
    uy[solid] = 0.0
    uz[solid] = 0.0
    rho = np.ones((cfg.nx, cfg.ny, cfg.nz), dtype=dtype)
    f = equilibrium(rho, ux.astype(dtype), uy.astype(dtype), uz.astype(dtype))
    return ChannelKernelState(f=f, config=cfg, solid=solid)


def run_channel(cfg: ChannelKernelConfig) -> dict[str, object]:
    """Run the channel and return the reduced statistics record.

    Returns a JSON-serialisable dict with the mean velocity profile, the four
    Reynolds-stress profiles, per-slab budget diagnostics and the wall-shear
    measurement from the bounce-back momentum exchange.
    """
    state = make_state(cfg)
    height = cfg.height
    wall_area = 2 * cfg.nx * cfg.nz
    moment_sum = np.zeros((7, cfg.ny))
    samples = 0
    diagnostics: list[dict[str, float]] = []
    for step_index in range(1, cfg.steps + 1):
        f_pre = state.f
        step(state)
        if step_index > cfg.warmup_steps and step_index % cfg.sample_interval == 0:
            rho, ux, uy, uz = macroscopic(state.f)
            moment_sum[0] += ux.mean(axis=(0, 2))
            moment_sum[1] += uy.mean(axis=(0, 2))
            moment_sum[2] += uz.mean(axis=(0, 2))
            moment_sum[3] += (ux * ux).mean(axis=(0, 2))
            moment_sum[4] += (uy * uy).mean(axis=(0, 2))
            moment_sum[5] += (uz * uz).mean(axis=(0, 2))
            moment_sum[6] += (ux * uy).mean(axis=(0, 2))
            samples += 1
        if step_index % 500 == 0 or step_index == cfg.steps:
            momentum = float(state.f[1].sum() - state.f[2].sum())
            # Momentum exchange on the lower wall: 2 * sum cx f_i over wall nodes.
            lower = state.f[:, :, 0, :]
            shear = 2.0 * float((C[:, 0, None, None] * lower).sum()) / wall_area
            diagnostics.append(
                {
                    "step": step_index,
                    "momentum_x": momentum,
                    "mass": float(state.f.sum()),
                    "measured_wall_shear": shear,
                    "measured_friction_velocity": float(np.sqrt(max(shear, 0.0))),
                }
            )
    mean = moment_sum / max(samples, 1)
    return {
        "schema": "tensorlbm-channel-kernel-numpy-v1",
        "configuration": {
            "nx": cfg.nx,
            "ny": cfg.ny,
            "nz": cfg.nz,
            "re_tau": cfg.re_tau,
            "u_tau": cfg.u_tau,
            "steps": cfg.steps,
            "warmup_steps": cfg.warmup_steps,
            "sample_interval": cfg.sample_interval,
            "collision": cfg.collision,
            "height": height,
            "nu": cfg.nu,
            "tau": cfg.tau,
            "dx_plus": cfg.dx_plus,
            "dz_plus": cfg.dx_plus,
            "streamwise_length_over_half_height": cfg.nx / (0.5 * height),
            "spanwise_length_over_half_height": cfg.nz / (0.5 * height),
            "body_force_acceleration": cfg.body_force_acceleration,
            "target_wall_shear": cfg.u_tau**2,
            "eddy_turnover_steps": height / cfg.u_tau,
        },
        "statistics": {
            "samples": samples,
            "statistics_steps": max(cfg.steps - cfg.warmup_steps, 0),
            "eddy_turnovers": max(cfg.steps - cfg.warmup_steps, 0) / (height / cfg.u_tau),
            "mean_velocity_profile": mean[0].tolist(),
            "crossflow_profiles": [mean[1].tolist(), mean[2].tolist()],
            "reynolds_stress_profiles_uu_vv_ww_uv": [mean[i].tolist() for i in range(3, 7)],
        },
        "diagnostics": diagnostics,
    }