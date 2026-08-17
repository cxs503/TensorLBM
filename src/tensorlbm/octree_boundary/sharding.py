"""Multi-device sharding of the octree boundary shell (``fine_devices``).

The shell leaves are the dominant compute of the hybrid architecture
(``f_leaf`` SoA — roughly 0.1-0.5M leaves x 19 channels).  :func:`shard_octree_shell`
partitions the leaf set in Morton order into contiguous shards and moves each
shard's leaf-local tensors (``f_leaf``, ``neighbor_table``, ``q_field``,
``bfl_mask``, ``leaf_level``, ...) to one fine device, while the L0/L1
hierarchy stays on the root device — the octree analogue of
``NestedStaticBlockAMR3D(fine_devices=...)``.

Per-root-step data flow (:func:`tensorlbm.octree_boundary.stepping.step_octree_shell_sharded`):

* root -> shard: the ghost values for the shard's interface links (small —
  interface links are a few percent of the leaves);
* shard -> shard: the post-collision populations of *cross-shard* neighbour
  and fan-out references only (the Morton cut surface, again a small
  fraction), through a precomputed request plan;
* shard -> root: the interface-transfer observation values and the BFL force
  link contributions (small), plus the full ``f_leaf`` once per root step for
  the restriction.

Numerical-equivalence contract: every state-affecting reduction (restriction,
reflux interface transfers, the observed MEM force) is assembled on the root
device in the *global* order of the unsharded run (Morton enum order for
leaves, interface-link row order for transfers, boundary-link enum order for
the force).  Cross-device transfers are exact copies.  The collision callback
may itself reduce over its spatial extent; CPU/GPU kernels can therefore differ
by a few ulps when shard sizes differ.  Regression tests require explicit
roundoff-level agreement, not a false bitwise-identity promise.

All torch constants (Morton indices, weights, the D3Q19 ``C``/``OPPOSITE``
arrays) are explicitly moved to the shard device at build time — never
defaulted to the host device.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tensorlbm.d3q19 import C, OPPOSITE
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import OPPOSITE as OPPOSITE27
from tensorlbm.octree_boundary.bfl import leaf_force_weights
from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT,
    FANOUT,
    REMOTE,
    SHELL_OUTSIDE,
    SOLID,
    OctreeGrid,
)
from tensorlbm.octree_boundary.stepping import (
    ShellGhostPlan,
    build_ghost_plan,
)


def split_leaf_bounds(n_leaf: int, n_shards: int) -> list[tuple[int, int]]:
    """Contiguous Morton-order leaf bounds ``[(lo, hi), ...]``, balanced by count."""
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    if n_shards > n_leaf:
        raise ValueError(
            f"more shards ({n_shards}) than leaves ({n_leaf}): empty shards "
            "would break the neighbor-table split",
        )
    base, extra = divmod(n_leaf, n_shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for s in range(n_shards):
        size = base + (1 if s < extra else 0)
        bounds.append((start, start + size))
        start += size
    assert start == n_leaf
    return bounds


def _slice_ghost_plan(
    plan: ShellGhostPlan, lo: int, hi: int, n_local: int,
    *,
    slot_device: torch.device | None = None,
) -> tuple[ShellGhostPlan, torch.Tensor]:
    """Subset of the global ghost plan whose leaves lie in ``[lo, hi)``.

    The returned plan's ``leaf`` field holds *local* leaf enums and its
    ``slot`` maps local leaves to local ghost rows (``-1`` = no ghost).  The
    donor arrays stay on the plan's device (the root device — they index the
    L1 parent field); ``slot`` is moved to ``slot_device`` (the shard device —
    it indexes the shard's ghost values).

    Returns ``(plan_slice, global_rows)`` where ``global_rows`` records each
    sliced row's position in the global (unsharded) ghost plan's row order —
    used by the sharded stepper to reassemble one global-order ghost fill.
    """
    device = plan.slot.device
    if plan.n_ghost == 0:
        slot = torch.full(
            (plan.slot.shape[0], n_local), -1, dtype=torch.int64, device=device,
        )
        if slot_device is not None:
            slot = slot.to(slot_device)
        return ShellGhostPlan(
            0,
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            torch.empty(0, dtype=torch.float64, device=device),
            slot,
        ), torch.empty(0, dtype=torch.int64, device=device)
    rows = torch.nonzero(
        (plan.leaf >= lo) & (plan.leaf < hi), as_tuple=False,
    ).squeeze(1)
    n_ghost = int(rows.shape[0])
    local_row = torch.full(
        (plan.n_ghost,), -1, dtype=torch.int64, device=device,
    )
    local_row[rows] = torch.arange(n_ghost, dtype=torch.int64, device=device)
    slot = plan.slot[:, lo:hi].clone()
    has = slot >= 0
    slot[has] = local_row[slot[has]]
    leaf = plan.leaf[rows] - lo
    if slot_device is not None:
        slot = slot.to(slot_device)
        # the leaf enums index the shard's leaf_level array (shard device);
        # all other donor arrays index the L1 parent field (root device)
        leaf = leaf.to(slot_device)
    return ShellGhostPlan(
        n_ghost=n_ghost,
        leaf=leaf,
        direction=plan.direction[rows],
        z0=plan.z0[rows], y0=plan.y0[rows], x0=plan.x0[rows],
        z1=plan.z1[rows], y1=plan.y1[rows], x1=plan.x1[rows],
        wz=plan.wz[rows], wy=plan.wy[rows], wx=plan.wx[rows],
        volume=plan.volume[rows],
        slot=slot,
        lev=plan.lev[rows] if plan.lev is not None else None,
    ), rows


def _rank_within_groups(labels: torch.Tensor) -> torch.Tensor:
    """For each row, its 0-based rank among rows sharing the same label."""
    rank = torch.zeros(labels.shape[0], dtype=torch.int64, device=labels.device)
    if labels.numel() == 0:
        return rank
    for d in range(int(labels.max().item()) + 1):
        sel = labels == d
        if bool(sel.any()):
            r = sel.cumsum(0) - 1
            rank[sel] = r[sel]
    return rank


@dataclass
class OctreeLeafShard:
    """One contiguous Morton-order slice of shell leaves owned by a fine device.

    Leaf-local tensors live on :attr:`device`.  The ``neighbor_table`` keeps
    global leaf enums for in-shard sources and rewrites cross-shard sources to
    :data:`REMOTE`; the per-substep post-collision values for those sources are
    fetched through :attr:`requests` into :attr:`remote_buf`, addressed by
    :attr:`remote_pos` (neighbours) and :attr:`fan_off`/:attr:`fan_len`
    (fan-out groups).

    The shard doubles as the per-shard octree facade passed to ``bfl_fn``:
    unknown attributes fall back to the root-side :class:`OctreeGrid`.
    """

    device: torch.device
    lo: int
    hi: int
    n_leaf: int
    f_leaf: torch.Tensor                 # (Q, n) populations dtype
    neighbor_table: torch.Tensor         # (Q, n) int64, REMOTE for cross-shard
    q_field: torch.Tensor                # (Q, n) float32
    bfl_mask: torch.Tensor               # (Q, n) bool
    leaf_level: torch.Tensor             # (n,) int64
    leaf_host_cell: torch.Tensor         # (n, 3) int64 (z, y, x) in L1 block
    leaf_volume: torch.Tensor            # (n,) float64
    leaf_force_weights: torch.Tensor     # (n,) float64
    opp: torch.Tensor                    # (Q,) int64 on device
    c_vec: torch.Tensor                  # (Q, 3) int64 on device
    ghost_plan: ShellGhostPlan           # donor arrays on the root device
    ghost_slot: torch.Tensor             # (Q, n) int64 local ghost row / -1
    remote_buf: torch.Tensor             # (R,) populations dtype on device
    remote_pos: torch.Tensor             # (Q, n) int64, -1 = in-shard source
    fan_off: torch.Tensor                # (Q, n) int64, -1 = no fan-out
    fan_len: torch.Tensor                # (Q, n) int64, 0 = none
    ghost_rows: torch.Tensor = field(    # (n_ghost_local,) int64 global row
        default_factory=lambda: torch.empty(0, dtype=torch.int64),
    )                                    # ranks in the global ghost plan
    requests: list = field(default_factory=list)
    # observation assembly (interface links of this shard, root-device ranks)
    link_rows: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    link_leaf: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    link_dir: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    out_rank: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    in_rank: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    # BFL force assembly: per direction d -> (local idx (n_d,), global rank (n_d,))
    bfl_links: dict = field(default_factory=dict)
    # per-substep working state (owned by the stepper)
    populations: torch.Tensor | None = None
    post_collision: torch.Tensor | None = None
    ghost_vals: torch.Tensor | None = None
    out: torch.Tensor | None = None
    remote_values: torch.Tensor | None = None   # facade alias of remote_buf
    # LES context: remote leaf macroscopic velocities are exchanged before
    # collision so WALE/Smagorinsky gradients remain continuous across a
    # Morton shard cut.  ``les_fan_members`` stores local member indices as
    # non-negative values and remote macro slots as ``-(slot+1)``.
    les_remote_pos: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int64),
    )
    les_remote_global: list[int] = field(default_factory=list)
    les_remote_centers: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 3)),
    )
    les_fan_members: dict = field(default_factory=dict)
    les_fan_distance: dict = field(default_factory=dict)
    les_requests: list = field(default_factory=list)
    les_neighbor_velocity: torch.Tensor | None = None
    les_neighbor_distance: torch.Tensor | None = None
    _octree: OctreeGrid | None = field(default=None, repr=False)

    # -- facade attributes used by bfl callbacks ---------------------------------
    @property
    def Q(self) -> int:  # noqa: N802
        return int(self.f_leaf.shape[0])

    @property
    def _opp(self) -> torch.Tensor:
        return self.opp

    @property
    def _c_vec(self) -> torch.Tensor:
        return self.c_vec

    @property
    def force_weights(self) -> torch.Tensor:
        return self.leaf_force_weights

    def __getattr__(self, name: str):
        # unknown attributes fall back to the root-side octree (d_max, meta,
        # interface_fanout, ...) — the shard is a faithful per-device facade.
        octree = self.__dict__.get("_octree")
        if octree is not None:
            return getattr(octree, name)
        raise AttributeError(name)


def shard_octree_shell(
    octree: OctreeGrid,
    fine_devices: list[torch.device | str],
    *,
    ghost_plan: ShellGhostPlan | None = None,
    solid_fallback: bool = True,
    l1_shape: tuple[int, int, int] | None = None,
) -> list[OctreeLeafShard]:
    """Split ``octree``'s leaves across ``fine_devices`` (one shard per device).

    The octree itself (and the L0/L1 hierarchy) stays on its own device — the
    root device.  ``ghost_plan`` may be a prebuilt global plan (built on the
    root device); otherwise one is built with ``solid_fallback``.
    """
    devices = [torch.device(d) for d in fine_devices]
    if not devices:
        raise ValueError("fine_devices must not be empty")
    lattice = octree.meta.get("lattice", "D3Q19")
    OPP = OPPOSITE27 if lattice == "D3Q27" else OPPOSITE
    if ghost_plan is None:
        shape = octree.meta["shape"] if l1_shape is None else l1_shape
        ghost_plan = build_ghost_plan(octree, shape, solid_fallback=solid_fallback)
    if ghost_plan.slot.device != octree.f_leaf.device:
        raise ValueError(
            "the ghost plan must live on the octree's (root) device — "
            "it indexes the L1 parent field",
        )

    n_leaf = octree.n_leaf
    Q = octree.Q
    bounds = split_leaf_bounds(n_leaf, len(devices))
    # shard index of every global leaf enum (bounds are contiguous)
    shard_of = torch.zeros(n_leaf, dtype=torch.int64)
    for s, (lo, hi) in enumerate(bounds):
        shard_of[lo:hi] = s
    shard_of = shard_of.to(octree.f_leaf.device)

    leaf_vol = octree.leaf_volume()
    fw = leaf_force_weights(octree)
    opp_root = octree._opp
    links = octree.interface_links                       # (n_link, 2) (i, d)
    n_link = int(links.shape[0])
    link_dir = links[:, 1]
    plan_dir = ghost_plan.direction if ghost_plan.n_ghost else link_dir
    out_rank = _rank_within_groups(link_dir) if n_link else link_dir
    in_rank = _rank_within_groups(plan_dir) if n_link else plan_dir
    rows_global = torch.arange(n_link, dtype=torch.int64)

    # fan-out groups: key (global leaf i, source row q) -> sorted member enums
    fan_groups: dict[tuple[int, int], list[int]] = {}
    for (i, q), members in octree.interface_fanout.items():
        fan_groups[(int(i), int(q))] = [int(m) for m in members]

    shards: list[OctreeLeafShard] = []
    for s, (lo, hi) in enumerate(bounds):
        dev = devices[s]
        n_s = hi - lo
        sl = slice(lo, hi)
        # leaf-local tensors on the fine device
        f_leaf = octree.f_leaf[:, sl].clone().to(dev)
        nt = octree.neighbor_table[:, sl].clone().to(dev)
        ghost_plan_s, ghost_rows_s = _slice_ghost_plan(
            ghost_plan, lo, hi, n_s, slot_device=dev,
        )
        shard = OctreeLeafShard(
            device=dev, lo=lo, hi=hi, n_leaf=n_s,
            f_leaf=f_leaf,
            neighbor_table=nt,
            q_field=octree.q_field[:, sl].clone().to(dev),
            bfl_mask=octree.bfl_mask[:, sl].clone().to(dev),
            leaf_level=octree.leaf_level[sl].clone().to(dev),
            leaf_host_cell=octree.leaf_host_cell[sl].clone().to(dev),
            leaf_volume=leaf_vol[sl].clone().to(dev),
            leaf_force_weights=fw[sl].clone().to(dev),
            opp=OPP.to(dev),
            c_vec=(C27 if lattice == "D3Q27" else C).to(dev),
            ghost_plan=ghost_plan_s,
            ghost_slot=ghost_plan_s.slot,
            ghost_rows=ghost_rows_s,
            remote_buf=torch.empty(0, dtype=f_leaf.dtype, device=dev),
            remote_pos=torch.full((Q, n_s), -1, dtype=torch.int64, device=dev),
            fan_off=torch.full((Q, n_s), -1, dtype=torch.int64, device=dev),
            fan_len=torch.zeros((Q, n_s), dtype=torch.int64, device=dev),
            requests=[],
            link_rows=torch.empty(0, dtype=torch.int64),
            link_leaf=torch.empty(0, dtype=torch.int64),
            link_dir=torch.empty(0, dtype=torch.int64),
            out_rank=torch.empty(0, dtype=torch.int64),
            in_rank=torch.empty(0, dtype=torch.int64),
            bfl_links={},
            _octree=octree,
        )
        # interface links owned by this shard (rows whose leaf is in-shard)
        if n_link:
            row_mask = (links[:, 0] >= lo) & (links[:, 0] < hi)
            rows_s = torch.nonzero(row_mask, as_tuple=False).squeeze(1)
            shard.link_rows = rows_s.clone()
            shard.link_leaf = (links[rows_s, 0] - lo).to(dev)
            shard.link_dir = links[rows_s, 1].clone()
            shard.out_rank = out_rank[rows_s].clone()
            shard.in_rank = in_rank[rows_s].clone()
        # BFL force assembly ranks (global enum order per direction)
        for d in range(1, Q):
            m = octree.bfl_mask[d, lo:hi]
            if not bool(m.any()):
                continue
            idx_local_root = torch.nonzero(m, as_tuple=False).squeeze(1)
            global_rank = octree.bfl_mask[d].cumsum(0) - 1
            shard.bfl_links[int(d)] = (
                idx_local_root.to(dev),
                global_rank[lo:hi][idx_local_root].clone(),
            )
        shards.append(shard)

    # ---- cross-shard request plans ------------------------------------------
    # Neighbour pulls: every (q, i) whose source leaf lives in another shard
    # needs ``populations_t[opp[q], src_local]`` once per substep.  Fan-out
    # groups are fetched member by member into a contiguous per-group region
    # of the receiving shard's remote_buf (member order = group order, so the
    # group mean uses the same member order as the unsharded gather).
    n_remote = torch.zeros(len(shards), dtype=torch.int64)
    n_fan = torch.zeros(len(shards), dtype=torch.int64)
    fan_bases: list[dict[tuple[int, int], int]] = [{} for _ in shards]
    for s, shard in enumerate(shards):
        lo, hi = shard.lo, shard.hi
        nt = octree.neighbor_table[:, lo:hi]
        for q in range(Q):
            src = nt[q]
            remote = (src >= 0) & ((src < lo) | (src >= hi))
            if bool(remote.any()):
                n_remote[s] += int(remote.sum().item())
        base = int(n_remote[s])
        for (i, q), members in fan_groups.items():
            if not (lo <= i < hi):
                continue
            fan_bases[s][(i, q)] = base
            base += len(members)
        n_fan[s] = base - int(n_remote[s])

    for s, shard in enumerate(shards):
        lo, hi = shard.lo, shard.hi
        n_s = hi - lo
        # neighbour fetches
        req: list[list[list[int]]] = []   # per target shard: [d_c, j_local, pos]
        for t in range(len(shards)):
            req.append([[], [], []])
        remote_pos = torch.full((Q, n_s), -1, dtype=torch.int64)
        fan_off = torch.full((Q, n_s), -1, dtype=torch.int64)
        fan_len = torch.zeros((Q, n_s), dtype=torch.int64)
        nt = octree.neighbor_table[:, lo:hi]
        n_remote_s = int(n_remote[s].item())
        slot = 0
        for q in range(Q):
            src = nt[q]
            for col in torch.nonzero(
                (src >= 0) & ((src < lo) | (src >= hi)),
                as_tuple=False,
            ).squeeze(1).tolist():
                j = int(src[col].item())
                t = int(shard_of[j].item())
                t_lo = bounds[t][0]
                req[t][0].append(int(OPP[q].item()))
                req[t][1].append(j - t_lo)
                req[t][2].append(slot)         # position in remote_buf
                remote_pos[q, col] = slot
                slot += 1
        assert slot == n_remote_s
        # fan-out fetches (consumer dir = opp[q], members in group order)
        fan_pos = n_remote_s
        for (i, q), members in fan_groups.items():
            if not (lo <= i < hi):
                continue
            i_local = i - lo
            fan_off[q, i_local] = fan_pos
            fan_len[q, i_local] = len(members)
            d_c = int(OPP[q].item())
            for k, m in enumerate(members):
                t = int(shard_of[m].item())
                t_lo = bounds[t][0]
                req[t][0].append(d_c)
                req[t][1].append(m - t_lo)
                req[t][2].append(fan_pos + k)
            fan_pos += len(members)
        assert fan_pos == n_remote_s + int(n_fan[s].item())
        remote_buf = torch.empty(
            n_remote_s + int(n_fan[s].item()),
            dtype=shard.f_leaf.dtype, device=shard.device,
        )
        requests: list = []
        for t in range(len(shards)):
            d_list, j_list, pos_list = req[t]
            if not d_list:
                continue
            d_idx = torch.tensor(d_list, dtype=torch.int64, device=shards[t].device)
            j_idx = torch.tensor(j_list, dtype=torch.int64, device=shards[t].device)
            out_pos = torch.tensor(pos_list, dtype=torch.int64, device=shard.device)
            requests.append((t, d_idx, j_idx, out_pos))
        # rewrite remote neighbour entries to the REMOTE sentinel and
        # in-shard entries to LOCAL enums (stream_gather and bfl_apply_gather
        # gather locally; cross-shard values resolve through remote_pos)
        rem = remote_pos >= 0
        shard.neighbor_table[rem] = torch.tensor(
            REMOTE, dtype=torch.int64, device=shard.device,
        )
        local = shard.neighbor_table >= 0
        shard.neighbor_table[local] = shard.neighbor_table[local] - lo
        shard.remote_pos = remote_pos.to(shard.device)
        shard.fan_off = fan_off.to(shard.device)
        shard.fan_len = fan_len.to(shard.device)
        shard.remote_buf = remote_buf
        shard.requests = requests

    # Build the independent, direction-free macro exchange used by the
    # sparse-leaf LES closure.  The streaming request plan above contains one
    # population per direction; a velocity gradient needs all Q populations
    # of every remote leaf, so it must use a deduplicated leaf plan.
    for s, shard in enumerate(shards):
        lo, hi = bounds[s]
        nt_global = octree.neighbor_table[:, lo:hi]
        remote_global: set[int] = set()
        for q in range(Q):
            for j in torch.nonzero(
                (nt_global[q] >= 0)
                & ((nt_global[q] < lo) | (nt_global[q] >= hi)),
                as_tuple=False,
            ).squeeze(1).tolist():
                remote_global.add(int(nt_global[q, j].item()))
        fan_specs: dict[tuple[int, int], list[int]] = {}
        for (i, d), members in fan_groups.items():
            if not (lo <= i < hi):
                continue
            key = (int(d), int(i - lo))
            fan_specs[key] = [int(m) for m in members]
            remote_global.update(
                m for m in members
                if not (lo <= int(m) < hi)
            )
        remote_list = sorted(remote_global)
        remote_slot = {g: k for k, g in enumerate(remote_list)}
        shard.les_remote_pos = torch.full(
            (Q, hi - lo), -1, dtype=torch.int64, device=shard.device,
        )
        for q in range(Q):
            src = nt_global[q]
            for j in torch.nonzero(
                (src >= 0) & ((src < lo) | (src >= hi)),
                as_tuple=False,
            ).squeeze(1).tolist():
                shard.les_remote_pos[q, j] = remote_slot[int(src[j].item())]
        fan_encoded: dict[tuple[int, int], list[int]] = {}
        for key, members in fan_specs.items():
            encoded: list[int] = []
            for m in members:
                if lo <= m < hi:
                    encoded.append(m - lo)
                else:
                    encoded.append(-remote_slot[m] - 1)
            fan_encoded[key] = encoded
        shard.les_fan_members = fan_encoded
        shard.les_remote_global = remote_list
        shard.les_remote_centers = octree.leaf_center[remote_list].to(
            device=shard.device,
        ) if remote_list else torch.empty(
            (0, 3), dtype=octree.leaf_center.dtype, device=shard.device,
        )
        fan_distance: dict[tuple[int, int], float] = {}
        for (d, i_local), members in fan_specs.items():
            if members:
                mean_center = octree.leaf_center[members].mean(dim=0)
                fan_distance[(d, i_local)] = float(
                    (octree.leaf_center[lo + i_local] - mean_center)
                    .abs().amax().item()
                )
        shard.les_fan_distance = fan_distance
        shard.les_remote_velocity = torch.empty(
            (3, len(remote_list)), dtype=shard.f_leaf.dtype,
            device=shard.device,
        )
        # Group remote leaves by owner shard.  The owner indices are local to
        # the source shard; receiver slots are local to this shard.
        for t, (t_lo, t_hi) in enumerate(bounds):
            pairs = [
                (g - t_lo, remote_slot[g])
                for g in remote_list if t_lo <= g < t_hi
            ]
            if pairs:
                shard.les_requests.append((
                    t,
                    torch.tensor(
                        [p[0] for p in pairs], dtype=torch.int64,
                        device=shards[t].device,
                    ),
                    torch.tensor(
                        [p[1] for p in pairs], dtype=torch.int64,
                        device=shard.device,
                    ),
                ))

    return shards


def shards_f_leaf(
    shards: list[OctreeLeafShard], device: torch.device,
) -> torch.Tensor:
    """Global (Morton-order) concatenation of the shard populations on ``device``."""
    return torch.cat(
        [shard.f_leaf.to(device=device) for shard in shards], dim=1,
    )


def refresh_octree_f_leaf(octree: OctreeGrid, shards: list[OctreeLeafShard]) -> None:
    """Mirror the shard states back into ``octree.f_leaf`` (root device).

    Called once per root step after the substep loop, so root-side observers
    (mass bookkeeping, finiteness checks, the example's joint-mass monitor)
    keep reading a live global view.
    """
    octree.f_leaf = shards_f_leaf(shards, octree.f_leaf.device)


def shards_all_finite(shards: list[OctreeLeafShard]) -> bool:
    return all(bool(torch.isfinite(shard.f_leaf).all()) for shard in shards)


__all__ = [
    "OctreeLeafShard",
    "refresh_octree_f_leaf",
    "shard_octree_shell",
    "shards_all_finite",
    "shards_f_leaf",
    "split_leaf_bounds",
]
