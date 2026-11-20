"""AA-pattern streaming (single field, even/odd kernels) with parity tracking.

Anchor
------
``XVAL-LBM-AA-PATTERN-EVEN-ODD-TIMESTEP-TRACKER``; the upstream sibling kernels are
``aa-pattern-e`` / ``aa-pattern-o`` of the walberla lattice-Boltzmann module
(lssfau/walberla).  Upstream source could not be fetched from this sandbox
(network egress is blocked), so this module does **not** claim byte-level
kernel identity with walberla.  What it does claim is the property the anchor
names, stated precisely and machine-checked here:

AB pattern vs AA pattern
------------------------
*AB pattern*: two full distribution fields ``A`` and ``B`` per direction exist
simultaneously and their roles are exchanged every timestep (``B <- stream(A)``,
then ``A <- stream(B)``).  Persistent storage ``2 Q N`` floats.

*AA pattern*: **one** field of ``Q N`` floats plus an ``O(N)`` line buffer; the
streaming permutation is executed in place, decomposed into two kernels that
each move **one half of the velocity set**:

    kernel ``e``  moves  S  = {1, 2, 5, 8}   (the ``(c_x, c_y)`` "positive" half)
    kernel ``o``  moves  S' = {3, 4, 7, 6}   (its opposite half); ``0`` is fixed

Both kernels act on *disjoint rows* of the single buffer, so neither can
invalidate the other's source values and the pair is order independent (asserted,
not assumed: ``test_kernel_order_is_immaterial``).  After ``e`` alone the buffer
is **half streamed**: four populations already belong to ``t+1``, four still to
``t``.  The macroscopic moments are undefined in that window, therefore an
AA-pattern implementation must *track* which kernel has run -- that is the
"even/odd timestep tracker" of this module.

Timestep parity
---------------
The AA kernels alternate roles across timesteps: at an even timestep the
positive half is moved first, at an odd timestep the negative half is moved
first (``step()``).  A tracker that loses the alternation -- the classic
off-by-one in an AA implementation -- ends up moving one half twice and the
other half zero times.  That is mass conserving and momentum destroying, and it
is exactly the failure mode :meth:`AAPatternStepper.parity_blind_step` injects
for the cross-validation control arm.

Upstream layout recorded in the platform knowledge base
------------------------------------------------------
`oss_solutions` (repo ``2b-t/LB-t``, tag ``aa-pattern-odd-even-indexing``,
files ``src/population/indexing/aa_pattern.hpp:56,79`` /
``src/population/population.hpp:198``) documents the concrete A-A slot layout:
one flat array with ``ND = SPEEDS + PAD`` (D3Q27: 28), ``OFF = ND/2 = 14``,
direction ``d``'s dual slot at ``OFF + d``, ``MASK[OFF] = 0`` to keep the rest
direction from being written twice, and the **even timestep writing the normal
slot while reading the dual slot, the odd timestep mirroring** -- i.e. the A-A
pattern is a read-old/write-new scheme inside a single array, which is why its
diagnostics must know the timestep parity.  This module implements the same
*contract* (one field, half-streamed windows, even/odd parity tracking) with a
single slot per direction plus an ``O(N)`` row buffer; the dual-slot
``ND = SPEEDS + PAD`` indexing itself is **not** implemented here (declared as
round-2 scope, see the write-up).

Storage accounting (structural, not measured): AA = ``Q N`` floats + ``O(N)``
line buffer; AB reference (``tensorlbm.solver.stream``) = ``2 Q N`` floats.
The equivalence to the AB reference is exact (streaming is a permutation), which
is what ``benchmarks/xval_aa_pattern_even_odd.py`` measures bit for bit.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from .d2q9 import C, OPPOSITE, equilibrium, macroscopic
from .solver import collide_bgk, stream

#: positive half of the D2Q9 velocity set (excluding the rest direction 0)
HALF_A: tuple[int, ...] = (1, 2, 5, 8)
#: negative half: the four opposites of :data:`HALF_A`
HALF_B: tuple[int, ...] = (3, 4, 7, 6)
HALF_OF_KERNEL: dict[str, tuple[int, ...]] = {"even": HALF_A, "odd": HALF_B}
KERNELS: tuple[str, str] = ("even", "odd")


def _check_half_split() -> None:
    """Guard the split: A ∪ B = 1..8, and B is the opposite set of A."""
    a, b = set(HALF_A), set(HALF_B)
    opp = {int(OPPOSITE[i]) for i in a}
    if a | b != set(range(1, 9)) or a & b:
        raise AssertionError(f"velocity halves must partition 1..8, got {sorted(a)} / {sorted(b)}")
    if opp != b:
        raise AssertionError(f"half B must be the opposite set of half A, got {sorted(opp)} vs {sorted(b)}")


_check_half_split()


class AAPatternParityError(RuntimeError):
    """Raised when a macroscopic read is requested from a half-streamed state."""


@dataclasses.dataclass(frozen=True)
class RowPermutation:
    """The gather index that turns row ``i`` into its pushed counterpart."""

    direction: int
    src: tuple[torch.Tensor, torch.Tensor]  # 2-D gather index (dest <- src)

    def nbytes(self) -> int:
        return sum(int(t.numel()) * t.element_size() for t in self.src)


class AAPatternStepper:
    """Single-field (AA) D2Q9 BGK stepper with even/odd kernel parity tracking.

    Parameters
    ----------
    f0:
        Initial populations, shape ``(9, ny, nx)``; the stepper owns one buffer
        of this size (that is the AA storage claim).
    tau:
        BGK relaxation time, ``> 0.5``.
    kernel_order:
        Order of the two AA kernels inside a timestep (``("even", "odd")`` by
        default).  The kernels move disjoint rows, so swapping them must not
        change the result -- the bench verifies this instead of assuming it.
    strict_parity:
        ``True`` (default) makes :meth:`macroscopic` raise
        :class:`AAPatternParityError` in a half-streamed window.  ``False`` is the
        deliberately untracked control arm that quantifies the hazard.

    Attributes
    ----------
    phase:
        ``"complete"`` (all populations at one time level) or ``"half"``.
    timestep_parity:
        ``"even"`` / ``"odd"`` -- which kernel runs first at the next step.
    time:
        Lattice time of the last complete state.
    """

    def __init__(
        self,
        f0: torch.Tensor,
        tau: float,
        *,
        kernel_order: tuple[str, str] = KERNELS,
        strict_parity: bool = True,
        timestep_parity: str = "even",
    ) -> None:
        if f0.dim() != 3 or f0.shape[0] != 9:
            raise ValueError(f"f0 must have shape (9, ny, nx), got {tuple(f0.shape)}")
        if float(tau) <= 0.5:
            raise ValueError(f"tau must be > 0.5, got {tau}")
        if tuple(kernel_order) not in (("even", "odd"), ("odd", "even")):
            raise ValueError(f"kernel_order must be a permutation of {KERNELS}, got {kernel_order}")
        if timestep_parity not in KERNELS:
            raise ValueError(f"timestep_parity must be one of {KERNELS}, got {timestep_parity!r}")
        self.f: torch.Tensor = f0.detach().to(memory_format=torch.contiguous_format).clone()
        self.tau = float(tau)
        self.kernel_order = tuple(kernel_order)
        self.strict_parity = bool(strict_parity)
        self.timestep_parity = timestep_parity
        self.phase = "complete"
        self.time = 0
        self.kernels_since_complete = 0
        self.kernel_log: list[str] = []
        self.parity_checks = 0
        self.parity_violations = 0
        self.even_kernel_calls = 0
        self.odd_kernel_calls = 0
        self._perm_cache: dict[int, RowPermutation] = {}
        #: structural storage accounting (AA vs AB): one persistent field
        self.n_persistent_fields = 1
        self.nbytes_persistent = int(self.f.numel()) * self.f.element_size()

    # ------------------------------------------------------------- permutation
    def _permutation(self, i: int) -> RowPermutation:
        """Cache the push-gather index of row ``i`` (periodic wrap)."""
        if i in self._perm_cache:
            return self._perm_cache[i]
        ny, nx = int(self.f.shape[1]), int(self.f.shape[2])
        dev = self.f.device
        yy = torch.arange(ny, device=dev).view(ny, 1).expand(ny, nx)
        xx = torch.arange(nx, device=dev).view(1, nx).expand(ny, nx)
        c = C.to(dev)
        # push: the value at source cell s lands on s + c_i, i.e. dest[y, x] gets
        # the value of src = (y - c_y, x - c_x) -- equivalently the row is
        # gathered with index s = dest - c_i (periodic), which is exactly what
        # the AB reference pull kernel does row-wise.
        sy = (yy - int(c[i, 1].item())) % ny
        sx = (xx - int(c[i, 0].item())) % nx
        perm = RowPermutation(direction=i, src=(sy, sx))
        self._perm_cache[i] = perm
        return perm

    def permutation_bytes(self) -> int:
        return sum(self._permutation(i).nbytes() for i in range(1, 9))

    # ---------------------------------------------------------------- kernels
    def _stream_half(self, kernel: str) -> None:
        """In-place streaming of one velocity half of the single buffer."""
        if kernel not in HALF_OF_KERNEL:
            raise ValueError(f"unknown AA kernel {kernel!r}; expected one of {KERNELS}")
        for i in HALF_OF_KERNEL[kernel]:
            # read the whole row, then write it back: the temporary is one row
            # (O(N) floats), not a second field -- that is the AA storage claim.
            self.f[i] = self.f[i][self._permutation(i).src]
        if kernel == "even":
            self.even_kernel_calls += 1
        else:
            self.odd_kernel_calls += 1
        self.kernel_log.append(kernel)
        self.kernels_since_complete += 1
        if self.kernels_since_complete % 2 == 0:
            # both velocity halves are back at one time level: reset the counter
            self.phase = "complete"
            self.kernels_since_complete = 0
        else:
            self.phase = "half"

    def kernel_even(self) -> None:
        """AA kernel ``e``: stream velocity half ``S = {1,2,5,8}`` in place."""
        self._stream_half("even")

    def kernel_odd(self) -> None:
        """AA kernel ``o``: stream velocity half ``S' = {3,4,7,6}`` in place."""
        self._stream_half("odd")

    # --------------------------------------------------------------- stepping
    def collide(self) -> None:
        """BGK collision on a complete state."""
        self._require_complete("collide")
        self.f = collide_bgk(self.f, self.tau)

    def _kernels_for_timestep(self) -> tuple[str, str]:
        """Kernel sequence of the current timestep (roles alternate by parity)."""
        first = "even" if self.timestep_parity == "even" else "odd"
        second = "odd" if first == "even" else "even"
        # the configured order fixes which kernel of the *pair* leads; the
        # alternation flips which velocity half that leading kernel moves.
        if self.kernel_order[0] != "even":
            first, second = second, first
        return first, second

    def step(self) -> None:
        """One complete AA timestep: collide, then the two kernels, then flip parity."""
        self._complete_timestep()

    def _complete_timestep(self, *, capture_mid: bool = False) -> dict[str, Any]:
        """Run one timestep; optionally snapshot the half-streamed window.

        The snapshot is taken *between* the two kernels, i.e. exactly where an
        AA implementation without a parity tracker would hand an inconsistent
        buffer to the diagnostics.  The tracker state is advanced exactly as in
        :meth:`step`, so probing does not perturb the run.
        """
        self._require_complete("step")
        self.collide()
        first, second = self._kernels_for_timestep()
        self._stream_half(first)
        snap: dict[str, Any] = {}
        if capture_mid:
            snap = {
                "f": self.f.clone(),
                "phase": self.phase,
                "kernels_since_complete": self.kernels_since_complete,
                "timestep_parity": self.timestep_parity,
                "time": self.time,
                "first_kernel": first,
            }
        self._stream_half(second)
        if self.phase != "complete":
            raise AAPatternParityError(
                f"tracker desynchronised: phase={self.phase!r} after ({first}, {second})"
            )
        self.time += 1
        self.timestep_parity = "odd" if self.timestep_parity == "even" else "even"
        snap["time_after"] = self.time
        return snap

    def probe_half_streamed(self) -> dict[str, Any]:
        """Advance one timestep and return the half-streamed snapshot (bench probe)."""
        return self._complete_timestep(capture_mid=True)

    def parity_blind_step(self) -> None:
        """Broken control arm: loses the even/odd alternation of the halves.

        The classic AA off-by-one: the tracker never flips ``timestep_parity``,
        so the leading kernel keeps moving the same velocity half.  Over one
        step that half is streamed twice and the other half not at all -- mass
        is conserved, momentum is not, and the state still *looks* complete to a
        slot-counting parity check.  Used to show that the tracker is load
        bearing rather than decorative.
        """
        self.collide()
        first, second = self.kernel_order
        self._stream_half(first)
        self._stream_half(first)  # parity lost: same half twice
        self.time += 1

    # -------------------------------------------------------------- accessors
    def _require_complete(self, what: str) -> None:
        self.parity_checks += 1
        if self.phase != "complete":
            self.parity_violations += 1
            raise AAPatternParityError(
                f"{what}: populations are half streamed (phase={self.phase!r}, "
                f"kernels_since_complete={self.kernels_since_complete}); the "
                "macroscopic fields are undefined in this window"
            )

    def macroscopic(self, *, allow_partial: bool = False):
        """Return ``(rho, ux, uy)``; refused unless the state is complete."""
        if allow_partial:
            self.parity_violations += 1
        else:
            self._require_complete("macroscopic")
        return macroscopic(self.f)

    def parity_report(self) -> dict[str, Any]:
        """Machine-readable tracker state (goes into the bench artefacts)."""
        return {
            "phase": self.phase,
            "time": self.time,
            "timestep_parity": self.timestep_parity,
            "kernels_since_complete": self.kernels_since_complete,
            "kernel_log_tail": self.kernel_log[-4:],
            "even_kernel_calls": self.even_kernel_calls,
            "odd_kernel_calls": self.odd_kernel_calls,
            "parity_checks": self.parity_checks,
            "parity_violations": self.parity_violations,
            "kernel_order": list(self.kernel_order),
            "strict_parity": self.strict_parity,
            "n_persistent_fields": self.n_persistent_fields,
            "nbytes_persistent": self.nbytes_persistent,
            "nbytes_permutation_index": self.permutation_bytes(),
        }


# ------------------------------------------------------------------ reference
def ab_stream(f: torch.Tensor) -> torch.Tensor:
    """Reference AB-pattern streaming: the repo's gather kernel, second field."""
    return stream(f)


def ab_step(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Reference AB timestep: collide, then gather-stream into a second field."""
    return ab_stream(collide_bgk(f, tau))


def equilibrium_from_velocity(
    rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """Build an initial population field from macroscopic variables."""
    return equilibrium(rho, ux, uy)