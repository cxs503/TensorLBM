"""L1 middle block: coarse -> L1 (2x) -> shell-leaf three-level hierarchy.

Stage 1 of ``docs/L1_MIDDLE_BLOCK_INTEGRATION_DESIGN.md`` (sections 3b/3d).
The L1 block is **replicated on every rank** (design Phase 1): the only new
collectives are the three chunked window ``all_gather``s (< 3MB/msg each);
the L1 stage itself performs zero communication and every operator
(cumulant collide, ``torch.roll`` streaming, ``torch.where`` freeze, stable
segmented reductions) is deterministic, so all ranks' L1 copies stay
bit-identical without any synchronization.

Layout
------
The persistent L1 tensor is ``(Q, nz_l1+2, ny_l1+2, nx_l1+2)`` — a one-cell
ghost ring, the same layout as ``StaticBlockAMR3D.fine_f``.  ``nz_l1 =
2*(box.z1-box.z0)`` etc.  The physical interior ``[:, 1:-1, 1:-1, 1:-1]`` is
the L1 field handed to the octree shell (``octree.meta["shape"]``); the
ghost ring is re-filled every substep from the time-lerped coarse window
(injection 2:1 + non-equilibrium rescale), so the block needs no far-field /
sponge / bounce-back of its own (it is strictly interior to the coarse
domain, design §3b).

Per root step::

    l1_phys_pre, posts_phys, posts_ghost = step_l1_block_distributed(
        block, coarse_window_old, coarse_window_new)
    # ... shell stage (see octree_integrated_validate.py):
    # step_octree_shell_distributed(octree, advance_shell, l1_phys_pre,
    #     block.physical_copy(), tau_coarse=block.tau_l1,
    #     l1_post=posts_phys, reflux=True, ...) -> mutates the copy
    block.set_physical(l1_f_phys)
    ledger = restrict_l1_block_to_coarse(block, coarse_window_new,
                                         coarse_window_post)
    write_window_back(coarse_f, coarse_window_new, block.win, in_slab, lo)

``posts_phys`` are the per-substep post-collision *physical* slices used by
the shell reflux observation (``l1_post`` list); ``posts_ghost`` are the
with-ghost post-collision states used by the coarse<->L1 box reflux
observation (``cell_volume = 1/8``, same convention as
``StaticBlockAMR3D.step``).

Freeze semantics
----------------
Solid L1 cells (``octree._solid``, L1 frame, mapped to the with-ghost grid)
are kept bitwise frozen across every substep: ``frozen = where(solid,
before, streamed)`` and the captured post-collision state is
``post_frozen = where(solid, before, post)`` (the state whose streaming
actually crosses the AMR interfaces).  This matches the design's
``torch.where(l1_solid_q, before, collided)`` and prevents the L1 from
evolving (or streaming through) the body interior; the wall force remains
the shell BFL's exclusive responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from tensorlbm.amr_population_transfer import rescale_nonequilibrium
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.fixed_nested_transfer import restrict_populations_2to1
from tensorlbm.kinetic_flux_register import (
    KineticInterfaceLinks,
    apply_face_local_reflux,
    build_kinetic_interface_links,
    observe_kinetic_interface_transfer,
)
from tensorlbm.octree_boundary.stepping import _tau_chain  # noqa: F401 (documented reuse)
from tensorlbm.refinement import BoxRegion
from tensorlbm.static_block_amr import (
    PopulationRefluxLedger,
    convective_refined_tau,
)

__all__ = [
    "L1BlockDistributed",
    "WindowInfo",
    "build_window_indices",
    "gather_window_chunked",
    "restrict_l1_block_to_coarse",
    "step_l1_block_distributed",
    "write_window_back",
]


@dataclass(frozen=True)
class WindowInfo:
    """The coarse box + 1-cell ring window (global coarse coordinates).

    ``cells`` is the ``(n_win, 3)`` int64 array of global ``(z, y, x)``
    window cells in row-major order (z-major, then y, then x).
    """

    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int
    cells: torch.Tensor

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z1 - self.z0 + 1, self.y1 - self.y0 + 1,
                self.x1 - self.x0 + 1)


def build_window_indices(
    domain_shape: tuple[int, int, int],
    box: BoxRegion,
    device: torch.device,
) -> WindowInfo:
    """Box + 1-cell ring cell set, clipped to the coarse domain.

    The ring must be complete on every side (the coarse reflux correction
    stencil and the L1 ghost donors both live in the ring), which requires
    the box to keep at least a 1-cell margin from the domain boundary —
    ``plan_body_shell_box`` with ``pad >= 1`` guarantees this.
    """
    nz, ny, nx = domain_shape
    z0 = max(0, box.z0 - 1)
    z1 = min(nz - 1, box.z1 + 1)
    y0 = max(0, box.y0 - 1)
    y1 = min(ny - 1, box.y1 + 1)
    x0 = max(0, box.x0 - 1)
    x1 = min(nx - 1, box.x1 + 1)
    if not (box.z0 > z0 and box.z1 < z1 and box.y0 > y0 and box.y1 < y1
            and box.x0 > x0 and box.x1 < x1):
        raise ValueError(
            "the L1 box must be strictly interior with a 1-cell coarse "
            f"margin; got box z:[{box.z0},{box.z1}) y:[{box.y0},{box.y1}) "
            f"x:[{box.x0},{box.x1}) in domain {domain_shape} — enlarge the "
            "domain or reduce --wake-cells/--shell-margin",
        )
    zz = torch.arange(z0, z1 + 1, device=device)
    yy = torch.arange(y0, y1 + 1, device=device)
    xx = torch.arange(x0, x1 + 1, device=device)
    gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
    cells = torch.stack((gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)),
                        dim=1)
    return WindowInfo(z0, z1, y0, y1, x0, x1, cells)


def gather_window_chunked(
    slab_field: torch.Tensor,
    win: WindowInfo,
    lo: int,
    hi: int,
    *,
    rank: int,
    world_size: int,
    max_bytes_per_msg: int = 3 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked all_gather of the box+ring window from per-rank x-slabs.

    ``slab_field`` is one rank's coarse slab ``(Q, nz, ny, nx_local+2)`` with
    one halo column per x side (global column ``g`` lives at local index
    ``g - lo + 1``).  Every window cell inside ``[lo, hi)`` contributes its
    real value; cells owned by other ranks contribute zero and the chunked
    all_gather sum assembles the full window.  All messages stay below
    ``max_bytes_per_msg`` (TCCL ~4MB deadlock guard).

    Returns ``(window (Q, n_win), in_slab (n_win,) bool)`` where
    ``in_slab`` marks the cells owned by this rank (reused by the write-back).
    """
    q = slab_field.shape[0]
    dtype = slab_field.dtype
    dev = slab_field.device
    wc = win.cells
    n_win = wc.shape[0]
    in_slab = (wc[:, 2] >= lo) & (wc[:, 2] < hi)
    local = torch.zeros(q, n_win, dtype=dtype, device=dev)
    if bool(in_slab.any()):
        zz = wc[in_slab, 0]
        yy = wc[in_slab, 1]
        xx = wc[in_slab, 2] - lo + 1
        local[:, in_slab] = slab_field[:, zz, yy, xx]
    full = torch.zeros(q, n_win, dtype=dtype, device=dev)
    chunk = max(1, int(max_bytes_per_msg
                       // (q * torch.finfo(dtype).bits // 8)))
    for c0 in range(0, n_win, chunk):
        c1 = min(c0 + chunk, n_win)
        piece = local[:, c0:c1].contiguous()
        gathered = [torch.empty_like(piece) for _ in range(world_size)]
        dist.all_gather(gathered, piece)
        for r in range(world_size):
            full[:, c0:c1] = full[:, c0:c1] + gathered[r]
    return full, in_slab


def write_window_back(
    coarse_f: torch.Tensor,
    window_patch: torch.Tensor,
    win: WindowInfo,
    in_slab: torch.Tensor,
    lo: int,
) -> None:
    """Write the corrected window patch back into this rank's coarse slab.

    Only the cells inside ``[lo, hi)`` are written (``in_slab`` mask);
    neighbour halo columns are refreshed by the next ``halo_exchange``.
    """
    wc = win.cells
    if bool(in_slab.any()):
        coarse_f[:, wc[in_slab, 0], wc[in_slab, 1],
                 wc[in_slab, 2] - lo + 1] = window_patch[:, in_slab]


class L1BlockDistributed:
    """Persistent L1 middle block (replicated per rank, Phase 1).

    Args:
        box: coarse-domain :class:`BoxRegion` of the L1 block (half-open
            ``[x0, x1)`` etc.).
        domain_shape: coarse domain ``(nz, ny, nx)`` (for the window frame).
        tau_coarse: coarse relaxation time; the L1 tau is the convective
            2:1 refinement of it.
        solid_l1: L1-frame boolean solid mask ``(nz_l1, ny_l1, nx_l1)``
            (``octree._solid``) used for the frozen-solid mask; ``None``
            disables freezing.
        collide_fn: ``collide_fn(f, tau) -> f_post`` applied to the whole
            with-ghost tensor (e.g. cumulant D3Q27).
        stream_fn: ``stream_fn(f_post) -> f_streamed`` — the 27-direction
            ``torch.roll`` pull-stream over the with-ghost tensor.
    """

    def __init__(
        self,
        box: BoxRegion,
        domain_shape: tuple[int, int, int],
        tau_coarse: float,
        *,
        q: int = 27,
        ratio: int = 2,
        ghost: int = 1,
        device: torch.device | str | None = None,
        solid_l1: torch.Tensor | None = None,
        collide_fn=None,
        stream_fn=None,
        maximum_reflux_correction_fraction: float = 0.2,
        correction_stencil: str = "exterior_cells",
    ) -> None:
        if ratio != 2:
            raise ValueError("the L1 block currently supports ratio=2 only")
        if ghost != 1:
            raise ValueError("the L1 block currently supports ghost=1 only")
        if q not in (19, 27):
            raise ValueError(f"unsupported lattice Q={q}")
        if collide_fn is None or stream_fn is None:
            raise TypeError("L1 block requires collide_fn and stream_fn")
        self.box = box
        self.ratio = ratio
        self.ghost = ghost
        self.q = q
        self.device = torch.device(
            "cpu" if device is None else device,
        )
        self.tau_coarse = float(tau_coarse)
        self.tau_l1 = convective_refined_tau(self.tau_coarse, self.ratio)
        self.collide_fn = collide_fn
        self.stream_fn = stream_fn
        self.maximum_reflux_correction_fraction = (
            maximum_reflux_correction_fraction
        )
        self.correction_stencil = correction_stencil

        g = ghost
        self.l1_shape = (
            (box.z1 - box.z0) * ratio,
            (box.y1 - box.y0) * ratio,
            (box.x1 - box.x0) * ratio,
        )
        nz_l1, ny_l1, nx_l1 = self.l1_shape
        self.l1_f = torch.zeros(
            (q, nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g),
            device=self.device,
        )

        # ---- window frame (box + 1-cell ring) ----
        self.win = build_window_indices(domain_shape, box, self.device)
        w = self.win
        nz_w, ny_w, nx_w = w.shape
        self.window_shape = (nz_w, ny_w, nx_w)

        # ---- ghost-layer mask + injection coarse-donor maps ----
        ghost_mask = torch.zeros(
            (nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g),
            dtype=torch.bool, device=self.device,
        )
        ghost_mask[0] = True
        ghost_mask[-1] = True
        ghost_mask[:, 0] = True
        ghost_mask[:, -1] = True
        ghost_mask[:, :, 0] = True
        ghost_mask[:, :, -1] = True
        self.ghost_mask = ghost_mask

        zoff = torch.arange(nz_l1 + 2 * g, device=self.device)
        yoff = torch.arange(ny_l1 + 2 * g, device=self.device)
        xoff = torch.arange(nx_l1 + 2 * g, device=self.device)
        zc = (box.z0 + (zoff - g) // ratio).clamp(w.z0, w.z1)
        yc = (box.y0 + (yoff - g) // ratio).clamp(w.y0, w.y1)
        xc = (box.x0 + (xoff - g) // ratio).clamp(w.x0, w.x1)
        self.zc_map = zc[:, None, None].expand(
            nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g,
        )
        self.yc_map = yc[None, :, None].expand(
            nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g,
        )
        self.xc_map = xc[None, None, :].expand(
            nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g,
        )

        # ---- frozen-solid mask (with-ghost frame) ----
        solid_q = torch.zeros(
            (nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g),
            dtype=torch.bool, device=self.device,
        )
        if solid_l1 is not None:
            if tuple(solid_l1.shape) != self.l1_shape:
                raise ValueError(
                    f"solid_l1 shape {tuple(solid_l1.shape)} must equal the "
                    f"L1 physical shape {self.l1_shape}",
                )
            solid_q[g:-g, g:-g, g:-g] = solid_l1.to(self.device)
        self.l1_solid_q = solid_q

        # ---- coarse<->L1 box interface links (window frame) ----
        box_owned = torch.zeros(
            (nz_w, ny_w, nx_w), dtype=torch.bool, device=self.device,
        )
        box_owned[
            box.z0 - w.z0: box.z1 - w.z0,
            box.y0 - w.y0: box.y1 - w.y0,
            box.x0 - w.x0: box.x1 - w.x0,
        ] = True
        self.box_links: KineticInterfaceLinks = build_kinetic_interface_links(
            box_owned, q=q,
        )

        # ---- L1 fine interface links (with-ghost frame) ----
        l1_owned = torch.zeros(
            (nz_l1 + 2 * g, ny_l1 + 2 * g, nx_l1 + 2 * g),
            dtype=torch.bool, device=self.device,
        )
        l1_owned[g:-g, g:-g, g:-g] = True
        self.l1_fine_links: KineticInterfaceLinks = (
            build_kinetic_interface_links(l1_owned, q=q)
        )

        self.l1_phys_pre: torch.Tensor | None = None
        self.l1_posts_phys: list[torch.Tensor] = []
        self.l1_posts_ghost: list[torch.Tensor] = []
        self.last_reflux: PopulationRefluxLedger | None = None

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def initialize_uniform(self, u_in: float = 0.06) -> None:
        """Uniform-inflow equilibrium init (bit-exact for a uniform start)."""
        shp = self.l1_f.shape[1:]
        rho = torch.ones(shp, device=self.device)
        ux = torch.full(shp, u_in, device=self.device)
        uy = torch.zeros(shp, device=self.device)
        uz = torch.zeros(shp, device=self.device)
        self.l1_f = equilibrium27(rho, ux, uy, uz)

    def initialize_from_window(self, coarse_window_new: torch.Tensor) -> None:
        """2x injection + neq rescale of the whole L1 grid from the window.

        ``_sample_parent_with_ghost`` semantics of ``StaticBlockAMR3D``
        (piecewise-constant parent sampling on the physical+ghost grid).
        """
        self.l1_f = self._sample_window(coarse_window_new)

    # ------------------------------------------------------------------
    # ghost fill / advance helpers
    # ------------------------------------------------------------------
    def _sample_window(self, parent_t: torch.Tensor) -> torch.Tensor:
        """Injection 2:1 sampling + neq rescale of the with-ghost L1 grid."""
        sampled = parent_t[:, self.zc_map, self.yc_map, self.xc_map]
        return rescale_nonequilibrium(
            sampled,
            tau_source=self.tau_coarse,
            tau_target=self.tau_l1,
            spatial_ratio=float(self.ratio),
        )

    def _fill_ghost(self, parent_t: torch.Tensor) -> None:
        sampled = self._sample_window(parent_t)
        self.l1_f = torch.where(self.ghost_mask, sampled, self.l1_f)

    def _advance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collide + stream + frozen-solid.  Returns ``(frozen, post_frozen)``."""
        before = self.l1_f
        post = self.collide_fn(before, self.tau_l1)
        post_frozen = torch.where(self.l1_solid_q, before, post)
        streamed = self.stream_fn(post_frozen)
        frozen = torch.where(self.l1_solid_q, before, streamed)
        return frozen, post_frozen

    # ------------------------------------------------------------------
    # physical-slice access (interface with the shell stepper)
    # ------------------------------------------------------------------
    def physical_slice(self) -> torch.Tensor:
        g = self.ghost
        return self.l1_f[:, g:-g, g:-g, g:-g]

    def physical_copy(self) -> torch.Tensor:
        """Contiguous copy of the physical interior.

        The distributed shell stepper mutates the tensor it is handed
        (restriction + reflux) and broadcasts it; passing a contiguous copy
        keeps the broadcast's flat-view write-back correct (a non-contiguous
        view's ``.contiguous()`` would detach the broadcast writes).
        """
        return self.physical_slice().contiguous()

    def set_physical(self, phys: torch.Tensor) -> None:
        g = self.ghost
        self.l1_f[:, g:-g, g:-g, g:-g] = phys

    # ------------------------------------------------------------------
    # root-step stage
    # ------------------------------------------------------------------
    def step(
        self,
        coarse_window_old: torch.Tensor,
        coarse_window_new: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Advance the L1 block by one root step (2 time-interpolated substeps).

        Returns ``(l1_phys_pre, posts_phys, posts_ghost)``:

        * ``l1_phys_pre`` — the physical L1 state at the root-step start
          (the shell stepper's time-lerp anchor ``l1_old``);
        * ``posts_phys`` — per-substep post-collision physical slices (the
          shell stepper's ``l1_post`` list);
        * ``posts_ghost`` — per-substep with-ghost post-collision states
          (the coarse<->L1 box reflux fine-transfer observations).

        The substep schedule mirrors ``StaticBlockAMR3D.step``: ghost is
        filled at ``alpha_start`` before the collide, and again at
        ``alpha_end`` after the stream, so the ghost layer always holds the
        time-lerped coarse state at the next substep's stream time.
        """
        g = self.ghost
        self.l1_phys_pre = self.physical_copy()
        posts_phys: list[torch.Tensor] = []
        posts_ghost: list[torch.Tensor] = []
        for s in range(self.ratio):
            alpha_start = s / self.ratio
            self._fill_ghost(torch.lerp(
                coarse_window_old, coarse_window_new, alpha_start,
            ))
            self.l1_f, post_frozen = self._advance()
            posts_phys.append(
                post_frozen[:, g:-g, g:-g, g:-g].contiguous(),
            )
            posts_ghost.append(post_frozen)
            alpha_end = (s + 1) / self.ratio
            self._fill_ghost(torch.lerp(
                coarse_window_old, coarse_window_new, alpha_end,
            ))
        self.l1_posts_phys = posts_phys
        self.l1_posts_ghost = posts_ghost
        return self.l1_phys_pre, posts_phys, posts_ghost

    def restrict_and_reflux(
        self,
        coarse_window_new: torch.Tensor,
        coarse_window_post: torch.Tensor,
    ) -> PopulationRefluxLedger:
        """L1 -> coarse restriction + box-interface kinetic reflux.

        Must run AFTER the shell stage (the physical interior already
        carries the shell restriction + shell reflux patch).  Writes the
        restricted box interior into ``coarse_window_new`` in place, then
        applies the face-local reflux correction on the 1-cell ring and
        returns the ledger (schema of ``StaticBlockAMR3D.step``).
        """
        g = self.ghost
        b = self.box
        w = self.win
        if not self.l1_posts_ghost:
            raise RuntimeError(
                "restrict_and_reflux requires the L1 substep post states "
                "(call step() first)",
            )
        l1_phys = self.physical_copy()
        restricted = restrict_populations_2to1(l1_phys)
        restricted = rescale_nonequilibrium(
            restricted,
            tau_source=self.tau_l1,
            tau_target=self.tau_coarse,
            spatial_ratio=1.0 / self.ratio,
        )
        coarse_window_new[
            :,
            b.z0 - w.z0: b.z1 - w.z0,
            b.y0 - w.y0: b.y1 - w.y0,
            b.x0 - w.x0: b.x1 - w.x0,
        ] = restricted
        coarse_transfer = observe_kinetic_interface_transfer(
            coarse_window_post, self.box_links,
        )
        fine_transfer = None
        for post_g in self.l1_posts_ghost:
            observed = observe_kinetic_interface_transfer(
                post_g, self.l1_fine_links,
                cell_volume=1.0 / self.ratio ** 3,
            )
            fine_transfer = (
                observed if fine_transfer is None
                else fine_transfer + observed
            )
        if fine_transfer is None:
            raise RuntimeError("L1 block omitted the fine interface transfer")
        coarse_window_new, report = apply_face_local_reflux(
            coarse_window_new,
            self.box_links,
            coarse_transfer,
            fine_transfer,
            maximum_correction_fraction=(
                self.maximum_reflux_correction_fraction
            ),
            correction_stencil=self.correction_stencil,
        )
        ledger = PopulationRefluxLedger(
            report.requested_inventory_correction,
            report.applied_inventory_correction,
            report.corrected_links,
            report.residual,
            report.limited_directions,
            report.raw_kinetic_mismatch,
            0.0, 1.0, 0.0, 1.0,
            report.maximum_applied_correction_fraction,
        )
        self.last_reflux = ledger
        return ledger


def step_l1_block_distributed(
    block: L1BlockDistributed,
    coarse_window_old: torch.Tensor,
    coarse_window_new: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Advance the L1 middle block by one root step (2 substeps)."""
    return block.step(coarse_window_old, coarse_window_new)


def restrict_l1_block_to_coarse(
    block: L1BlockDistributed,
    coarse_window_new: torch.Tensor,
    coarse_window_post: torch.Tensor,
) -> PopulationRefluxLedger:
    """L1 -> coarse restriction + box reflux (see
    :meth:`L1BlockDistributed.restrict_and_reflux`)."""
    return block.restrict_and_reflux(coarse_window_new, coarse_window_post)
