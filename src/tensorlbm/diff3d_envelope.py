"""Memory-bounded 3D D3Q19 expanded-form AD: segmented replay checkpointing.

``tensorlbm.autograd_path.rollout(checkpoint=True)`` keeps the *discrete*
unrolled graph differentiable but its activation memory still grows with the
number of steps (non-reentrant ``torch.utils.checkpoint`` stores the input of
every step).  For a 256^3 D3Q19 lattice one distribution function is
``19 * 256^3 * 4 B = 1.28 GiB`` (fp32), so a few hundred stored states blow any
practical envelope.  This module adds the missing cost control:

* :class:`SegmentStore` -- a disk-backed store of boundary states
  (``numpy.memmap``, exact dtype, no quantisation) so the replay states do not
  occupy RAM;
* :func:`segmented_rollout` -- an ``autograd.Function`` that runs the forward
  rollout under ``no_grad`` (writing one boundary state per segment to disk)
  and, in the backward pass, replays the segments in reverse order from the
  stored states, reconstructing the graph one segment at a time.  Peak
  activation memory is therefore ``O(seg_len)`` + a few fields instead of
  ``O(n_steps)``, at the price of one extra forward (replay) -- the classic
  Revolve/treeverse trade-off, here with the states on disk;
* :func:`peak_rss_gib` -- VmHWM based peak-RSS probe (the honest
  process-level memory number for a CPU run, no allocator counters);
* initial conditions (:func:`divergence_free_ic`, :func:`taylor_green_2d_ic`,
  :func:`shear_wave_ic`) plus their analytic references
  (:func:`taylor_green_energy`, :func:`shear_wave_amplitude`) used by the
  envelope benchmark.

Everything is exact discrete differentiation of the forward code (expanded-form
/ BPTT autograd, not a continuous adjoint), so the gradient it returns is the
same quantity a full-graph backward returns -- :func:`segmented_rollout` is
verified against the full-graph path and against central finite differences in
``tests/test_diff3d_envelope.py``.
"""

from __future__ import annotations

import math
import mmap
import os
import shutil
from collections.abc import Callable

import numpy as np
import torch

StepFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# peak memory probe
# --------------------------------------------------------------------------- #
def peak_rss_gib() -> float:
    """Process peak resident set size (``VmHWM``) in GiB.

    ``VmHWM`` is the kernel high-water mark of the RSS, i.e. the true peak the
    process ever reached -- the number a memory envelope is judged by.  It is
    *not* resettable, so every measurement must run in a fresh subprocess.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 2**20  # kB -> GiB
    except OSError:  # pragma: no cover - non-Linux
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# --------------------------------------------------------------------------- #
# disk-backed boundary states
# --------------------------------------------------------------------------- #
class SegmentStore:
    """Disk-backed store of rollout boundary states (exact dtype, memmap)."""

    def __init__(self, root: str, n_segments: int, shape, dtype: torch.dtype):
        self.root = root
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.np_dtype = np.dtype(torch.empty(0, dtype=dtype).numpy().dtype)
        os.makedirs(root, exist_ok=True)
        self.paths = [os.path.join(root, f"seg_{i:05d}.npy") for i in range(n_segments)]
        self._maps = [
            np.lib.format.open_memmap(p, mode="w+", dtype=self.np_dtype, shape=self.shape)
            for p in self.paths
        ]

    # -- io ---------------------------------------------------------------- #
    # A memmap write/read goes through the page cache, and *dirty/cached pages
    # of a memmap are charged to the process RSS* -- at 256^3 a single state is
    # 1.28 GiB, so a naive store would report an RSS that is really page cache.
    # Every access therefore ends with flush + MADV_DONTNEED so the kernel
    # evicts the pages again and VmHWM stays a true working-set number.
    def _evict(self, i: int) -> None:
        arr = self._maps[i]
        arr.flush()
        try:
            arr._mmap.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):  # pragma: no cover - non-Linux
            pass

    def write(self, i: int, tensor: torch.Tensor) -> None:
        self._maps[i][...] = tensor.detach().to("cpu").numpy()
        self._evict(i)

    def read(self, i: int) -> torch.Tensor:
        """Load segment ``i`` into a fresh RAM tensor (never a memmap view)."""
        out = torch.from_numpy(np.array(self._maps[i]))
        self._evict(i)
        return out

    # -- housekeeping ------------------------------------------------------ #
    @property
    def bytes_per_state(self) -> int:
        return int(np.prod(self.shape)) * self.np_dtype.itemsize

    def size_gib(self) -> float:
        return len(self.paths) * self.bytes_per_state / 2**30

    def close(self, remove: bool = True) -> None:
        for arr in self._maps:
            del arr
        self._maps = []
        if remove:
            shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# segmented (disk-offloaded) unrolled autograd
# --------------------------------------------------------------------------- #
class _SegmentedRollout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, tau, n_steps, seg_len, step_fn, store, inner_ckpt):  # type: ignore[override]
        n_seg = math.ceil(n_steps / seg_len)
        f = f0.detach().clone()
        with torch.no_grad():
            for s in range(n_seg):
                store.write(s, f)
                n = min(seg_len, n_steps - s * seg_len)
                for _ in range(n):
                    f = step_fn(f, tau.detach())
        ctx.n_steps = n_steps
        ctx.seg_len = seg_len
        ctx.n_seg = n_seg
        ctx.step_fn = step_fn
        ctx.store = store
        ctx.inner_ckpt = inner_ckpt
        ctx.save_for_backward(f0, tau)
        return f

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        f0, tau = ctx.saved_tensors
        need_f0, need_tau = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_f0 = torch.zeros_like(f0) if need_f0 else None
        grad_tau = torch.zeros_like(tau) if (need_tau and torch.is_tensor(tau)) else None
        g = grad_out
        with torch.enable_grad():
            for s in reversed(range(ctx.n_seg)):
                n = min(ctx.seg_len, ctx.n_steps - s * ctx.seg_len)
                fin = ctx.store.read(s).clone().requires_grad_(True)
                t_local = tau.detach().clone().requires_grad_(True)
                f = fin
                for _ in range(n):
                    if ctx.inner_ckpt:
                        f = torch.utils.checkpoint.checkpoint(
                            ctx.step_fn, f, t_local, use_reentrant=False
                        )
                    else:
                        f = ctx.step_fn(f, t_local)
                # ``autograd.backward`` (not ``.grad``) so a reentrant
                # checkpointed replay is allowed to run in this pass.
                torch.autograd.backward([f], [g])
                gf, gt = fin.grad, t_local.grad
                if s > 0:
                    g = gf
                elif grad_f0 is not None:  # gradient w.r.t. the initial condition
                    grad_f0 = gf
                if grad_tau is not None and gt is not None:
                    grad_tau = grad_tau + gt
        return grad_f0, grad_tau, None, None, None, None, None


def segmented_rollout(
    f0: torch.Tensor,
    tau: torch.Tensor | float,
    n_steps: int,
    step_fn: StepFn,
    *,
    seg_len: int = 1,
    store: SegmentStore | None = None,
    inner_ckpt: bool = False,
) -> torch.Tensor:
    """Roll ``n_steps`` of ``step_fn`` keeping an exact graph at bounded memory.

    Args:
        f0: initial distribution ``(19, nz, ny, nx)``.
        tau: relaxation time (graph-connected 0-dim tensor if it is inverted).
        n_steps: number of steps.
        step_fn: ``(f, tau) -> f``; must accept a 0-dim tensor ``tau``.
        seg_len: steps per stored boundary state -- the memory/latency knob
            (RAM scales with ``seg_len``, disk with ``ceil(n_steps/seg_len)``).
        store: optional pre-allocated :class:`SegmentStore`; a temporary one
            under ``tempfile.mkdtemp`` is created (and removed) otherwise.
        inner_ckpt: wrap each replayed step in ``torch.utils.checkpoint``
            (non-reentrant; keeps the replay tape at one step for
            ``seg_len > 1``).

    Returns:
        Final distribution with ``grad_fn`` connected to ``f0`` and ``tau``.
    """
    if seg_len < 1:
        raise ValueError("seg_len must be >= 1")
    if not torch.is_tensor(tau):
        tau = torch.tensor(float(tau), dtype=f0.dtype)
    own = store is None
    if own:
        import tempfile

        root = tempfile.mkdtemp(prefix="lbm3d_segstore_")
        store = SegmentStore(root, math.ceil(n_steps / seg_len), f0.shape, f0.dtype)
    assert store is not None
    try:
        if not (tau.requires_grad or f0.requires_grad):
            f = f0
            with torch.no_grad():
                for _ in range(n_steps):
                    f = step_fn(f, tau)
            return f
        out = _SegmentedRollout.apply(f0, tau, int(n_steps), int(seg_len), step_fn, store, inner_ckpt)
        if own:
            # the store must stay alive until the backward replay is done, so
            # it is tied to the lifetime of the output tensor
            import weakref

            weakref.finalize(out, store.close, True)
        return out
    finally:
        pass


# --------------------------------------------------------------------------- #
# initial conditions
# --------------------------------------------------------------------------- #
def divergence_free_ic(
    n: int,
    *,
    seed: int = 0,
    amplitude: float = 0.05,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random smooth divergence-free velocity field, ``(3, n, n, n)``.

    The field is a discrete curl ``u = rot(psi)`` of a seeded random vector
    potential, evaluated spectrally (``U = i k x Psi``).  ``div rot psi = 0``
    holds *mode by mode*, so the field is divergence-free to machine precision
    for any (even) resolution -- unlike a plain Leray projection, whose
    Hermitian symmetry is broken on the Nyquist plane when ``n`` is even.
    The field is rescaled to ``amplitude`` (RMS ~ amplitude) and has zero net
    momentum.  Unlike the 2D Taylor-Green field it excites all 19 D3Q19
    directions and all three axes, so it is the gradient-correctness IC.
    """
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(3, n, n, n, generator=g, dtype=torch.float64)
    k = torch.fft.fftfreq(n, d=1.0 / (2 * torch.pi)).double()
    # zero the Nyquist wavenumber: fftfreq reports -n/2 there, which breaks the
    # odd symmetry k(-m) = -k(m) and hence the Hermitian symmetry of the curl
    # spectrum (the real-part truncation would then corrupt div-free-ness).
    k[n // 2] = 0.0
    kmat = torch.stack(torch.meshgrid(k, k, k, indexing="ij"))  # (3,n,n,n)
    P = torch.fft.fftn(psi, dim=(1, 2, 3))
    U = torch.stack(
        [
            1j * (kmat[1] * P[2] - kmat[2] * P[1]),
            1j * (kmat[2] * P[0] - kmat[0] * P[2]),
            1j * (kmat[0] * P[1] - kmat[1] * P[0]),
        ]
    )
    u = torch.fft.ifftn(U, dim=(1, 2, 3)).real
    u = u - u.mean(dim=(1, 2, 3), keepdim=True)
    rms = u.square().mean().sqrt()
    return (u / rms * amplitude).to(dtype)


def taylor_green_2d_ic(n: int, *, amplitude: float = 0.05, dtype=torch.float32) -> torch.Tensor:
    """2D Taylor-Green vortex embedded in a 3D box, ``(3, n, n, n)``.

    ``u_x = A sin x cos y``, ``u_y = -A cos x sin y``, ``u_z = 0`` (Taylor 1937;
    the classic LBM validation, e.g. Hou et al. 1995, JCP 118:329).  It is an
    exact solution of the incompressible Navier-Stokes equations with kinetic
    energy ``E(t) = E_0 exp(-4 nu t)`` (see :func:`taylor_green_energy`).
    """
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * (2 * torch.pi / n)
    # tensor layout is (component, z, y, x): the *first* meshgrid axis is z
    z, y, x = torch.meshgrid(t, t, t, indexing="ij")
    assert z.shape == (n, n, n)
    u = torch.stack(
        [
            amplitude * torch.sin(x) * torch.cos(y),
            -amplitude * torch.cos(x) * torch.sin(y),
            torch.zeros_like(x),
        ]
    )
    return u.to(dtype)


def taylor_green_energy(
    t: float, *, amplitude: float = 0.05, nu: float, k: float = 1.0
) -> float:
    """Analytic kinetic energy of the 2D Taylor-Green vortex (per unit volume).

    ``u = A e^{-nu (2 k^2) t} (sin kx cos ky, -cos kx sin ky, 0)`` is an exact
    solution of the 2D incompressible Navier-Stokes equations (the convective
    term is a pure gradient absorbed by the pressure), hence
    ``E(t) = E0 exp(-4 nu k^2 t)`` with ``E0 = A^2/4`` (Taylor 1937; Hou et al.,
    J. Comput. Phys. 118:329, 1995 -- the standard LBM validation).
    """
    return 0.25 * amplitude**2 * math.exp(-4.0 * nu * k * k * t)


def shear_wave_ic(
    n: int, *, amplitude: float = 0.05, wavelength: int = 1, dtype=torch.float32
) -> torch.Tensor:
    """Transverse shear wave along z, ``u_x = A sin(2 pi wavelength z / n)``.

    Viscous decay ``A(t) = A_0 exp(-nu k^2 t)``, ``k = 2 pi wavelength / n``
    (:func:`shear_wave_amplitude`).  Only the z axis is excited, so it isolates
    the z-streaming part of the discrete adjoint.
    """
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * (2 * torch.pi / n)
    # transverse by construction: u_x varies along z (axis 0), not along x
    z, _, _ = torch.meshgrid(t, t, t, indexing="ij")
    u = torch.stack(
        [
            amplitude * torch.sin(wavelength * z),
            torch.zeros_like(z),
            torch.zeros_like(z),
        ]
    )
    return u.to(dtype)


def shear_wave_amplitude(t: float, *, amplitude: float = 0.05, nu: float, k: float) -> float:
    """Analytic amplitude of the decaying shear wave."""
    return amplitude * math.exp(-nu * k * k * t)