"""Turbulence statistics for post-processing LBM time series.

PowerFlow and XFlow both provide turbulence statistics dashboards covering:
Reynolds stresses, turbulence kinetic energy (TKE), turbulence intensity (Tu),
integral length scales, and higher-order moments (skewness, flatness).

This module provides equivalent functionality for TensorLBM, operating on
either:

* A 2-D ``(ny, nx)`` or 3-D ``(nz, ny, nx)`` time-averaged snapshot captured
  by :class:`tensorlbm.postprocess.RunningStats`, **or**
* A list of checkpoint tensors loaded from a completed job's run directory.

Main classes / functions
------------------------
:class:`TurbulenceStatsAccumulator`
    Incremental accumulator for running computation of first- and second-order
    statistics from a time series.  Compatible with the existing
    :class:`tensorlbm.postprocess.RunningStats` but extends it with cross-
    correlations (Reynolds stresses) and higher-order moments.

:func:`compute_reynolds_stresses`
    Compute all six independent components of the Reynolds stress tensor
    ``R_ij = <u_i' u_j'>`` from pre-computed mean and RMS fields.

:func:`compute_turbulence_intensity`
    Compute the turbulence intensity ``Tu = sqrt(2k/3) / U_ref`` where
    ``k = (uu + vv + ww) / 2`` is the TKE.

:func:`compute_turbulence_length_scale`
    Estimate the integral length scale from the autocorrelation of a
    velocity signal (Taylor's frozen-turbulence hypothesis).

:func:`turbulence_stats_from_checkpoints`
    High-level function that loads all checkpoints from a run directory,
    accumulates statistics, and returns a summary dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Incremental accumulator
# ---------------------------------------------------------------------------

@dataclass
class TurbulenceStatsAccumulator:
    """Incremental first- and second-order turbulence statistics accumulator.

    Call :meth:`update` with each new velocity snapshot; retrieve statistics
    at any time via the property accessors.

    Supports both 2-D (ux, uy only) and 3-D (ux, uy, uz) velocity fields.

    Args:
        is_3d: If ``True``, expects three velocity components.  If ``False``
            (default), only ``ux`` and ``uy`` are tracked.
    """

    is_3d: bool = False
    _n: int = field(default=0, init=False, repr=False)
    _sum_u: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_v: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_w: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_uu: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_vv: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_ww: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_uv: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_uw: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_vw: torch.Tensor | None = field(default=None, init=False, repr=False)
    # Third- and fourth-order moments for skewness/flatness of u
    _sum_u3: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sum_u4: torch.Tensor | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of velocity snapshots accumulated so far."""
        return self._n

    def update(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one velocity snapshot.

        Args:
            ux: Streamwise velocity field.
            uy: Wall-normal (or y) velocity field.
            uz: Spanwise velocity field (required if ``is_3d=True``).
        """
        if self._sum_u is None:
            self._sum_u = torch.zeros_like(ux)
            self._sum_v = torch.zeros_like(uy)
            self._sum_uu = torch.zeros_like(ux)
            self._sum_vv = torch.zeros_like(uy)
            self._sum_uv = torch.zeros_like(ux)
            self._sum_u3 = torch.zeros_like(ux)
            self._sum_u4 = torch.zeros_like(ux)
            if self.is_3d and uz is not None:
                self._sum_w = torch.zeros_like(uz)
                self._sum_ww = torch.zeros_like(uz)
                self._sum_uw = torch.zeros_like(ux)
                self._sum_vw = torch.zeros_like(uy)

        self._n += 1
        self._sum_u = self._sum_u + ux
        self._sum_v = self._sum_v + uy
        self._sum_uu = self._sum_uu + ux * ux
        self._sum_vv = self._sum_vv + uy * uy
        self._sum_uv = self._sum_uv + ux * uy
        self._sum_u3 = self._sum_u3 + ux**3
        self._sum_u4 = self._sum_u4 + ux**4
        if self.is_3d and uz is not None and self._sum_w is not None:
            self._sum_w = self._sum_w + uz
            self._sum_ww = self._sum_ww + uz * uz
            self._sum_uw = self._sum_uw + ux * uz
            self._sum_vw = self._sum_vw + uy * uz

    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self._n == 0 or self._sum_u is None:
            raise RuntimeError("No samples accumulated yet; call update() first.")

    @property
    def mean_u(self) -> torch.Tensor:
        """Time-averaged streamwise velocity <U>."""
        self._check_ready()
        return self._sum_u / self._n  # type: ignore[operator]

    @property
    def mean_v(self) -> torch.Tensor:
        """Time-averaged y-velocity <V>."""
        self._check_ready()
        return self._sum_v / self._n  # type: ignore[operator]

    @property
    def mean_w(self) -> torch.Tensor | None:
        """Time-averaged z-velocity <W>; ``None`` in 2-D mode."""
        if not self.is_3d or self._sum_w is None:
            return None
        return self._sum_w / self._n

    @property
    def uu(self) -> torch.Tensor:
        """Reynolds normal stress <u'u'> = <UU> − <U>²."""
        self._check_ready()
        return self._sum_uu / self._n - self.mean_u**2  # type: ignore[operator]

    @property
    def vv(self) -> torch.Tensor:
        """Reynolds normal stress <v'v'>."""
        self._check_ready()
        return self._sum_vv / self._n - self.mean_v**2  # type: ignore[operator]

    @property
    def ww(self) -> torch.Tensor | None:
        """Reynolds normal stress <w'w'>; ``None`` in 2-D mode."""
        if not self.is_3d or self._sum_w is None:
            return None
        return self._sum_ww / self._n - self.mean_w**2  # type: ignore[operator]

    @property
    def uv(self) -> torch.Tensor:
        """Reynolds shear stress <u'v'>."""
        self._check_ready()
        return self._sum_uv / self._n - self.mean_u * self.mean_v  # type: ignore[operator]

    @property
    def uw(self) -> torch.Tensor | None:
        """Reynolds shear stress <u'w'>; ``None`` in 2-D mode."""
        if not self.is_3d or self._sum_uw is None or self._sum_w is None:
            return None
        return self._sum_uw / self._n - self.mean_u * self.mean_w

    @property
    def vw(self) -> torch.Tensor | None:
        """Reynolds shear stress <v'w'>; ``None`` in 2-D mode."""
        if not self.is_3d or self._sum_vw is None or self._sum_w is None:
            return None
        return self._sum_vw / self._n - self.mean_v * self.mean_w

    @property
    def tke(self) -> torch.Tensor:
        """Turbulence kinetic energy k = (uu + vv + ww) / 2."""
        self._check_ready()
        k = 0.5 * (self.uu + self.vv)
        if self.is_3d and self.ww is not None:
            k = k + 0.5 * self.ww
        return k

    @property
    def skewness_u(self) -> torch.Tensor:
        """Skewness S = <u'³> / <u'²>^(3/2) of streamwise fluctuations."""
        self._check_ready()
        mu = self.mean_u
        uu_r = self.uu.clamp(min=1e-20)
        m3 = self._sum_u3 / self._n - 3.0 * mu * uu_r - mu**3  # type: ignore[operator]
        return m3 / uu_r**(1.5)

    @property
    def flatness_u(self) -> torch.Tensor:
        """Flatness (kurtosis) F = <u'⁴> / <u'²>² of streamwise fluctuations."""
        self._check_ready()
        mu = self.mean_u
        uu_r = self.uu.clamp(min=1e-20)
        m4 = self._sum_u4 / self._n - 4.0 * mu * (self._sum_u3 / self._n) + \
             6.0 * mu**2 * uu_r + 3.0 * mu**4  # type: ignore[operator]
        return m4 / uu_r**2

    def to_dict(self) -> dict[str, Any]:
        """Serialise statistics to a plain dict of Python scalars / lists.

        Returns a flat dict suitable for JSON serialisation via the REST API.
        """
        self._check_ready()

        def _to_list(t: torch.Tensor | None) -> list | None:
            return t.cpu().tolist() if t is not None else None

        return {
            "n_samples": self._n,
            "mean_u": _to_list(self.mean_u),
            "mean_v": _to_list(self.mean_v),
            "mean_w": _to_list(self.mean_w),
            "uu": _to_list(self.uu),
            "vv": _to_list(self.vv),
            "ww": _to_list(self.ww),
            "uv": _to_list(self.uv),
            "uw": _to_list(self.uw),
            "vw": _to_list(self.vw),
            "tke": _to_list(self.tke),
            "skewness_u": _to_list(self.skewness_u),
            "flatness_u": _to_list(self.flatness_u),
        }

    def reset(self) -> None:
        """Reset accumulator to zero state."""
        self._n = 0
        self._sum_u = self._sum_v = self._sum_w = None
        self._sum_uu = self._sum_vv = self._sum_ww = None
        self._sum_uv = self._sum_uw = self._sum_vw = None
        self._sum_u3 = self._sum_u4 = None


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def compute_reynolds_stresses(
    ux_mean: torch.Tensor,
    uy_mean: torch.Tensor,
    ux_rms: torch.Tensor,
    uy_rms: torch.Tensor,
    uz_mean: torch.Tensor | None = None,
    uz_rms: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Reynolds stresses from time-averaged and RMS velocity fields.

    This convenience function assumes the RMS fields were computed from
    fluctuations so that ``uu = ux_rms²``.

    Args:
        ux_mean: Time-averaged x-velocity.
        uy_mean: Time-averaged y-velocity.
        ux_rms: Root-mean-square of x-velocity fluctuations.
        uy_rms: Root-mean-square of y-velocity fluctuations.
        uz_mean: Time-averaged z-velocity (3-D only).
        uz_rms: Root-mean-square of z-velocity fluctuations (3-D only).

    Returns:
        Dict with keys ``"uu"``, ``"vv"``, ``"ww"`` (None if 2-D),
        ``"tke"``, ``"tu_percent"``.
    """
    uu = ux_rms**2
    vv = uy_rms**2
    ww = uz_rms**2 if uz_rms is not None else None

    k = 0.5 * (uu + vv + (ww if ww is not None else torch.zeros_like(uu)))
    u_ref = torch.sqrt(ux_mean**2 + uy_mean**2 +
                       ((uz_mean**2) if uz_mean is not None else torch.zeros_like(ux_mean)))
    u_ref = u_ref.clamp(min=1e-12)
    tu = torch.sqrt(2.0 * k / 3.0) / u_ref * 100.0  # percent

    return {"uu": uu, "vv": vv, "ww": ww, "tke": k, "tu_percent": tu}


def compute_turbulence_intensity(
    tke: torch.Tensor,
    u_ref: float,
) -> torch.Tensor:
    """Compute turbulence intensity ``Tu = sqrt(2k/3) / U_ref`` (%).

    Args:
        tke: Turbulence kinetic energy field.
        u_ref: Reference velocity magnitude.

    Returns:
        Turbulence intensity in percent.
    """
    return torch.sqrt(2.0 * tke / 3.0) / max(u_ref, 1e-12) * 100.0


def compute_turbulence_length_scale(
    signal: torch.Tensor,
    dt: float = 1.0,
    u_conv: float = 1.0,
    max_lag: int | None = None,
) -> float:
    """Estimate the integral length scale via autocorrelation (Taylor's hypothesis).

    .. math::

        L = U_{conv} \\int_0^{\\infty} R_{uu}(\\tau) \\, d\\tau

    where :math:`R_{uu}(\\tau)` is the normalised autocorrelation of the
    velocity signal, and the integral is truncated at the first zero-crossing.

    Args:
        signal: 1-D velocity time series ``u(t)`` of length ``N``.
        dt: Time step between samples (lattice units; default 1).
        u_conv: Convection velocity for Taylor's frozen-turbulence hypothesis
            (default 1.0 → temporal scale only).
        max_lag: Maximum lag to consider (default: ``N // 4``).

    Returns:
        Estimated integral length scale in lattice units.
    """
    n = signal.numel()
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 1)

    sig = signal.float()
    mu = sig.mean()
    sig_c = sig - mu
    var = (sig_c**2).mean().item()
    if var < 1e-20:
        return 0.0

    # Biased autocorrelation via FFT (efficient)
    nfft = 2 * n
    sig_pad = torch.zeros(nfft, dtype=torch.float32, device=signal.device)
    sig_pad[:n] = sig_c
    S = torch.fft.rfft(sig_pad)
    R_full = torch.fft.irfft(S * S.conj())[:n].real / (n * var)

    # Integrate to first zero-crossing
    integral = 0.0
    for lag in range(1, max_lag + 1):
        r = float(R_full[lag].item())
        if r <= 0.0:
            break
        integral += r * dt

    return integral * u_conv


# ---------------------------------------------------------------------------
# High-level: compute statistics from a job run directory
# ---------------------------------------------------------------------------

def turbulence_stats_from_checkpoints(
    run_dir: str | Path,
    is_3d: bool = False,
    max_checkpoints: int = 50,
) -> dict[str, Any]:
    """Load checkpoints from a run directory and compute turbulence statistics.

    Iterates through all available ``checkpoint_f.pt`` files, recovering the
    velocity field at each step and feeding it into a
    :class:`TurbulenceStatsAccumulator`.

    Args:
        run_dir: Path to the job output directory.
        is_3d: Set ``True`` for 3-D jobs (D3Q19 / D3Q27 checkpoints).
        max_checkpoints: Limit the number of checkpoints processed (most
            recent *max_checkpoints* are used when there are more).

    Returns:
        Dict with keys:

        * ``"n_samples"`` – number of checkpoints processed
        * ``"mean_u"``, ``"mean_v"``, ``"mean_w"`` – time-averaged velocities
          (flat lists)
        * ``"uu"``, ``"vv"``, ``"ww"`` – Reynolds normal stresses
        * ``"uv"`` – Reynolds shear stress
        * ``"tke"`` – turbulence kinetic energy
        * ``"tu_percent"`` – turbulence intensity (%)
        * ``"skewness_u"``, ``"flatness_u"`` – higher-order moments
    """
    from tensorlbm import load_checkpoint

    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.rglob("checkpoint_f.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return {"error": "No checkpoints found", "n_samples": 0}

    # Use the most recent max_checkpoints only
    if len(ckpts) > max_checkpoints:
        ckpts = ckpts[-max_checkpoints:]

    acc = TurbulenceStatsAccumulator(is_3d=is_3d)

    for ckpt in ckpts:
        try:
            f, _step, _meta = load_checkpoint(ckpt.parent)
        except Exception:
            continue

        if is_3d:
            if f.shape[0] == 27:
                from tensorlbm.d3q27 import macroscopic27 as macro
            else:
                from tensorlbm.d3q19 import macroscopic3d as macro  # type: ignore[assignment]
            _rho, ux, uy, uz = macro(f)
            acc.update(ux, uy, uz)
        else:
            from tensorlbm.d2q9 import macroscopic
            _rho, ux, uy = macroscopic(f)
            acc.update(ux, uy)

    if acc.count == 0:
        return {"error": "No valid checkpoints loaded", "n_samples": 0}

    result = acc.to_dict()

    # Add turbulence intensity summary scalar (domain mean)
    tke_t = acc.tke
    mean_u_t = acc.mean_u
    mean_v_t = acc.mean_v
    mean_w_t = acc.mean_w
    u_mag = torch.sqrt(
        mean_u_t**2 + mean_v_t**2 +
        (mean_w_t**2 if mean_w_t is not None else torch.zeros_like(mean_u_t))
    )
    tu = compute_turbulence_intensity(tke_t, float(u_mag.mean().item()))
    result["tu_mean_percent"] = float(tu.mean().item())
    result["tke_mean"] = float(tke_t.mean().item())

    return result


__all__ = [
    "TurbulenceStatsAccumulator",
    "compute_reynolds_stresses",
    "compute_turbulence_intensity",
    "compute_turbulence_length_scale",
    "turbulence_stats_from_checkpoints",
]
