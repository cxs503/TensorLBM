"""Topology of the octree shell: neighbour table and cross-level registry.

This module builds the two pieces the Octree-LBM-solver reference project
*does not* implement — same-level Morton neighbour lookup *and* explicit
cross-level (1:2) interface links:

* **donor** (fine -> coarse): a depth-2 leaf whose direction ``d`` exits its
  depth-1 parent cell records the coarse depth-1 leaf it lands on;
* **fanout** (coarse -> fine): a depth-1 leaf pointing at a refined depth-2
  region records the set of wall-facing depth-2 sub-leaves (4 for face
  directions, 2 for edge directions of D3Q19), marked ``FANOUT`` in the
  neighbour table and enumerated in ``interface_fanout``;
* **interface links**: every ``(leaf i, direction d)`` whose neighbour leaves
  the shell (``SHELL_OUTSIDE``) is registered in ``interface_links`` — the
  shell side of the shell<->L1-block exchange (design doc §3.2).

Topology checks verify the P1 acceptance criteria: neighbour-table symmetry
(same-level reciprocity plus donor<->fanout one-to-one correspondence) and
2:1 balance (leaf-neighbour depth difference <= 1).
"""
from __future__ import annotations

import torch

from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT,
    FANOUT,
    SHELL_OUTSIDE,
    SOLID,
    morton_encode_batch,
)


def _classify_targets(
    target: torch.Tensor,
    lvl: int,
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
) -> torch.Tensor:
    """Classify out-of-tree neighbour coordinates.

    ``target`` holds Morton lattice coordinates at depth ``lvl`` (``(n, 3)``,
    columns ``(x, y, z)``).  Returns per-row one of
    ``DOMAIN_OUT / SOLID / SHELL_OUTSIDE``.
    """
    nz, ny, nx = shape
    bound = torch.tensor(
        [nx, ny, nz], dtype=torch.int64, device=target.device,
    ) << lvl
    out_of_bounds = ((target < 0) | (target >= bound)).any(dim=1)
    world = (target.to(torch.float64) + 0.5) / (2 ** lvl)
    dist2 = (
        (world[:, 0] - center[0]) ** 2
        + (world[:, 1] - center[1]) ** 2
        + (world[:, 2] - center[2]) ** 2
    )
    inside = dist2 <= radius ** 2
    return torch.where(
        out_of_bounds,
        torch.full_like(out_of_bounds, DOMAIN_OUT, dtype=torch.int64),
        torch.where(
            inside,
            torch.full_like(inside, SOLID, dtype=torch.int64),
            torch.full_like(inside, SHELL_OUTSIDE, dtype=torch.int64),
        ),
    )


def build_neighbor_table(grid) -> None:
    """Fill ``grid.neighbor_table`` in place (Q x n_leaf int64)."""
    shape = grid.meta["shape"]
    center = grid.meta["center"]
    radius = grid.meta["radius"]
    k = grid._k
    device = grid.leaf_morton.device
    n1 = grid.n_leaf_level(1)
    n2 = grid.n_leaf_level(2)
    n_leaf = grid.n_leaf
    l1_coords = grid._l1_coords
    l2_coords = grid._l2_coords
    opp = grid._opp
    c_vec = grid._c_vec

    # sorted Morton lookup tables (position -> leaf enum)
    l1_morton = morton_encode_batch(
        torch.full((n1,), 1, dtype=torch.int64), l1_coords, k,
    )
    l1_sorted, l1_order = torch.sort(l1_morton)
    if n2:
        l2_morton = morton_encode_batch(
            torch.full((n2,), 2, dtype=torch.int64), l2_coords, k,
        )
        l2_sorted, l2_order = torch.sort(l2_morton)
        # refined depth-1 parent coordinates (parents of depth-2 leaves)
        refined_parents = torch.unique(l2_coords >> 1, dim=0)
        ref_morton = morton_encode_batch(
            torch.full((refined_parents.shape[0],), 1, dtype=torch.int64),
            refined_parents, k,
        )
        ref_sorted, _ = torch.sort(ref_morton)
    else:
        l2_sorted = torch.empty(0, dtype=torch.int64, device=device)
        l2_order = torch.empty(0, dtype=torch.int64, device=device)
        ref_sorted = torch.empty(0, dtype=torch.int64, device=device)

    nt = torch.full((grid.Q, n_leaf), SHELL_OUTSIDE, dtype=torch.int64, device=device)
    nt[0] = torch.arange(n_leaf, dtype=torch.int64, device=device)  # rest -> self
    donor = torch.full((grid.Q, n_leaf), -1, dtype=torch.int64, device=device)

    def _hit(sorted_m: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """searchsorted membership: returns (hit_mask, order_positions)."""
        if sorted_m.shape[0] == 0:
            return torch.zeros(q.shape[0], dtype=torch.bool, device=q.device), \
                torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
        p = torch.searchsorted(sorted_m, q)
        pm = p.clamp(max=sorted_m.shape[0] - 1)
        hit = (p < sorted_m.shape[0]) & (sorted_m[pm] == q)
        return hit, p

    for d in range(1, grid.Q):
        c_d = c_vec[d]
        cd = torch.tensor(
            [int(c_d[0]), int(c_d[1]), int(c_d[2])],
            dtype=torch.int64, device=device,
        )
        od = int(opp[d].item())
        # Morton lattice bounds per level (columns (x, y, z)):
        #   level 1 -> [0, 2*nx) x [0, 2*ny) x [0, 2*nz)
        #   level 2 -> [0, 4*nx) x [0, 4*ny) x [0, 4*nz)
        # Out-of-range targets must never be resolved through the Morton
        # lookups: a negative coordinate encodes as two's-complement garbage
        # (all lower bits set) and a coordinate >= 2^(k+level) has its high
        # bits silently truncated by the interleave, so either can collide
        # with the code of a legitimate leaf far away.  Zero the hits and let
        # ``_classify_targets`` (geometric, bounds-aware) decide instead.
        bound1 = torch.tensor(
            [shape[2], shape[1], shape[0]], dtype=torch.int64, device=device,
        ) << 1
        bound2 = bound1 << 1

        # ---- depth-2 leaves (first: donor links define the fanout groups) --
        if n2:
            i2 = torch.arange(n2, dtype=torch.int64, device=device) + n1
            target2 = l2_coords + cd                       # (n2, 3)
            inb2 = ((target2 >= 0) & (target2 < bound2)).all(dim=1)
            parent = target2 >> 1                          # depth-1 coord
            q_parent = morton_encode_batch(
                torch.full((n2,), 1, dtype=torch.int64), parent, k,
            )
            hit_p, p_p = _hit(l1_sorted, q_parent)
            q2 = morton_encode_batch(
                torch.full((n2,), 2, dtype=torch.int64), target2, k,
            )
            hit2, p2 = _hit(l2_sorted, q2)
            hit_p = hit_p & inb2
            hit2 = hit2 & inb2
            nbr2 = torch.full((n2,), SHELL_OUTSIDE, dtype=torch.int64, device=device)
            nbr2 = torch.where(
                hit_p,
                l1_order[p_p.clamp(max=l1_sorted.shape[0] - 1)],
                nbr2,
            )
            donor[d, i2] = torch.where(hit_p, nbr2, torch.full_like(nbr2, -1))
            nbr2 = torch.where(
                hit2,
                n1 + l2_order[p2.clamp(max=l2_sorted.shape[0] - 1)],
                torch.where(
                    hit_p,
                    nbr2,
                    _classify_targets(target2, 2, shape, center, radius),
                ),
            )
            nt[d, i2] = nbr2

            # fanout groups are the *reverse* of the donor links: a fine leaf i
            # streaming along d onto coarse leaf j implies j fans out along
            # opp[d] onto i.  Reverse-collecting the donors makes the
            # cross-level registry symmetric by construction (the fine-leaf
            # link endpoint and the coarse-leaf reverse endpoint do not
            # coincide for diagonal D3Q19 links, so a purely geometric
            # face-facing definition cannot be one-to-one).
            if bool(hit_p.any()):
                donor_idx = torch.nonzero(hit_p, as_tuple=False).squeeze(1)
                for ii in donor_idx.tolist():
                    j_leaf = int(nbr2[ii].item())
                    grid._fanout_groups.setdefault(
                        (j_leaf, od), [],
                    ).append(int(i2[ii].item()))

        # ---- depth-1 leaves ------------------------------------------------
        if n1:
            target1 = l1_coords + cd                       # (n1, 3)
            inb1 = ((target1 >= 0) & (target1 < bound1)).all(dim=1)
            q1 = morton_encode_batch(
                torch.full((n1,), 1, dtype=torch.int64), target1, k,
            )
            hit1, p1 = _hit(l1_sorted, q1)
            hit_ref, p_ref = _hit(ref_sorted, q1)
            hit1 = hit1 & inb1
            hit_ref = hit_ref & inb1
            # FANOUT only when the reverse donor registry is non-empty; an
            # empty (all-solid wall-facing) refined neighbour is solid.
            has_fo = torch.zeros(n1, dtype=torch.bool, device=device)
            for pos in torch.nonzero(hit_ref, as_tuple=False).squeeze(1).tolist():
                has_fo[pos] = bool(
                    grid._fanout_groups.get((int(pos), int(d)), []),
                )
            nbr1 = torch.where(
                hit1,
                l1_order[p1.clamp(max=l1_sorted.shape[0] - 1)],
                torch.where(
                    hit_ref & has_fo,
                    torch.full_like(hit1, FANOUT, dtype=torch.int64),
                    torch.where(
                        hit_ref & ~has_fo,
                        torch.full_like(hit1, SOLID, dtype=torch.int64),
                        _classify_targets(target1, 1, shape, center, radius),
                    ),
                ),
            )
            nt[d, :n1] = nbr1

    grid.neighbor_table = nt
    grid.cross_level_donor = donor


def build_interface_registry(grid) -> None:
    """Fill ``grid.interface_links`` and ``grid.interface_fanout``.

    * ``interface_links``: every ``(leaf i, direction d)`` with
      ``neighbor_table[d, i] == SHELL_OUTSIDE`` (link leaves the shell).
    * ``interface_fanout``: coarse->fine groups collected during neighbour
      construction (``{(i, d): [leaf enums]}``).
    """
    nt = grid.neighbor_table
    links = torch.nonzero(nt == SHELL_OUTSIDE, as_tuple=False)  # (n, 2) (d, i)
    if links.shape[0]:
        links = links[:, [1, 0]].contiguous()                  # (i, d)
    grid.interface_links = links.to(torch.int64)
    grid.interface_fanout = dict(grid._fanout_groups)


# ---------------------------------------------------------------------------
# Topology checks (P1 acceptance)
# ---------------------------------------------------------------------------


def check_neighbor_symmetry(grid) -> dict:
    """Neighbour-table reciprocity (same level) + donor/fanout one-to-one.

    Returns a dict with the counts of violated links; ``"symmetric": True``
    when nothing is violated.
    """
    nt = grid.neighbor_table
    opp = grid._opp
    level = grid.leaf_level
    n_leaf = grid.n_leaf
    device = nt.device
    opp_idx = opp.to(device)

    viol_same = 0
    viol_donor = 0
    viol_fanout = 0

    for d in range(1, grid.Q):
        od = int(opp_idx[d].item())
        nbr = nt[d]
        nbr_opp = nt[od]
        same = (nbr >= 0) & (level == level[nbr.clamp(min=0)])
        # same-level reciprocity
        if bool(same.any()):
            j = nbr[same]
            back = nbr_opp[j]
            viol_same += int((back != torch.nonzero(same, as_tuple=False).squeeze(1)).sum().item())
        # cross-level donor: fine -> coarse must appear in the coarse fanout
        donor_links = (nbr >= 0) & (level[nbr.clamp(min=0)] < level)
        if bool(donor_links.any()):
            idx = torch.nonzero(donor_links, as_tuple=False).squeeze(1)
            for i in idx.tolist():
                j = int(nbr[i].item())
                fg = grid.interface_fanout.get((j, od))
                if fg is None or int(i) not in fg:
                    viol_donor += 1
        # fanout: coarse -> fine must be the fine leaf's donor
        fo = (nbr == FANOUT)
        if bool(fo.any()):
            idx = torch.nonzero(fo, as_tuple=False).squeeze(1)
            for i in idx.tolist():
                for k in grid.interface_fanout.get((int(i), int(d)), []):
                    if int(nt[od, k].item()) != int(i):
                        viol_fanout += 1

    ok = viol_same == 0 and viol_donor == 0 and viol_fanout == 0
    return {
        "symmetric": ok,
        "viol_same_level": int(viol_same),
        "viol_donor": int(viol_donor),
        "viol_fanout": int(viol_fanout),
    }


def check_balance_21(grid) -> dict:
    """2:1 balance — any leaf-neighbour depth difference is <= 1.

    Neighbours of depth-1 leaves are depth-1 (same level) or depth-2 fanout
    sub-leaves; neighbours of depth-2 leaves are depth-2 (same level) or a
    depth-1 donor.  The buffer/transition band guarantees no depth-2 leaf
    ever faces an unrefined block cell inside the shell.
    """
    nt = grid.neighbor_table
    level = grid.leaf_level
    n_leaf = grid.n_leaf
    viol = 0
    for d in range(1, grid.Q):
        nbr = nt[d]
        valid = (nbr >= 0)
        if not bool(valid.any()):
            continue
        i = torch.nonzero(valid, as_tuple=False).squeeze(1)
        j = nbr[valid]
        diff = (level[j] - level[i]).abs()
        viol += int((diff > 1).sum().item())
    return {"balanced_21": viol == 0, "violations": int(viol)}


def check_no_dangling(grid) -> dict:
    """No dangling leaves: every leaf has at least one fluid/shell link."""
    nt = grid.neighbor_table
    n_leaf = grid.n_leaf
    fluid_links = (nt[1:] >= 0) | (nt[1:] == FANOUT)
    dangling = int((~fluid_links.any(dim=0)).sum().item())
    return {"no_dangling": dangling == 0, "dangling_leaves": int(dangling)}


def check_interface_links(grid) -> dict:
    """Interface-link registry is complete and lands inside the L1 block."""
    nt = grid.neighbor_table
    links = grid.interface_links
    expected = int((nt == SHELL_OUTSIDE).sum().item())
    complete = links.shape[0] == expected
    host = grid.leaf_host_cell[links[:, 0]] if links.shape[0] else torch.empty(
        (0, 3), dtype=torch.int64,
    )
    nz, ny, nx = grid.meta["shape"]
    in_block = (
        (host[:, 0] >= 0) & (host[:, 0] < nz)
        & (host[:, 1] >= 0) & (host[:, 1] < ny)
        & (host[:, 2] >= 0) & (host[:, 2] < nx)
    ) if links.shape[0] else torch.tensor([], dtype=torch.bool)
    return {
        "complete": complete,
        "n_links": int(links.shape[0]),
        "all_hosts_in_block": bool(in_block.all().item()) if links.shape[0] else True,
    }


def run_topology_checks(grid) -> dict:
    """All P1 topology checks in one dict."""
    checks = {
        "symmetry": check_neighbor_symmetry(grid),
        "balance_21": check_balance_21(grid),
        "no_dangling": check_no_dangling(grid),
        "interface_links": check_interface_links(grid),
    }
    return checks
