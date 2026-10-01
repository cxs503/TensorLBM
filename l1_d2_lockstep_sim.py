"""16-rank lockstep CPU simulation of the L1 d_max=2 root step.

Each rank runs in its own thread with its own data shard; the fake
torch.distributed exchanges REAL per-rank data through a threading barrier,
so the collective pattern must be rank-symmetric — any rank-divergence
deadlocks on the barrier (exactly like TCCL). One root step, both interleave
modes. No GPU.
"""
import sys
sys.path.insert(0, "/root/TensorLBM_feat2/src")
import copy
import threading
import time
import types

import torch

WORLD = 16
INTERLEAVE = True          # try True first
D_MAX = 2

# ---------------- correct fake dist (barrier-lockstep) ----------------
_lock = threading.Lock()
_barrier = threading.Barrier(WORLD)
_token = [0]
_store = {}                 # token -> {rank: tensor}
_cur_rank = [0]             # thread-local rank (set under lock per thread)
_fail = [None]


def _next_token():
    with _lock:
        t = _token[0]
        _token[0] += 1
        return t


def _check(name, tok):
    if _fail[0] is not None:
        raise RuntimeError(f"earlier failure: {_fail[0]}")


def _ag(gathered, piece):
    r = _cur_rank[0]
    tok = _next_token()
    with _lock:
        _store.setdefault(tok, {})[r] = piece.clone()
    try:
        _barrier.wait(timeout=300)
    except threading.BrokenBarrierError:
        _fail[0] = f"all_gather token={tok} rank={r} BROKEN BARRIER (rank divergence)"
        raise
    with _lock:
        pieces = dict(_store[tok])
        # NOTE: keep the entry (no del) — other threads still read it
    for k in range(WORLD):
        gathered[k] = pieces[k]
    if tok % 50 == 0:
        print(f"[ag] token={tok} rank={r} piece={tuple(piece.shape)} "
              f"{piece.numel()*piece.element_size()/1e6:.2f}MB", flush=True)


def _bc(t, src=0):
    r = _cur_rank[0]
    tok = _next_token()
    with _lock:
        _store.setdefault(tok, {})[r] = t.clone()
    try:
        _barrier.wait(timeout=300)
    except threading.BrokenBarrierError:
        _fail[0] = f"broadcast token={tok} rank={r} BROKEN BARRIER (rank divergence)"
        raise
    with _lock:
        pieces = dict(_store[tok])
        del _store[tok]
    t.copy_(pieces[src])
    if tok % 50 == 0:
        print(f"[bc] token={tok} rank={r} {tuple(t.shape)} "
              f"{t.numel()*t.element_size()/1e6:.2f}MB", flush=True)


def _ar(t, op=None):
    r = _cur_rank[0]
    tok = _next_token()
    with _lock:
        _store.setdefault(tok, {})[r] = t.clone()
    try:
        _barrier.wait(timeout=300)
    except threading.BrokenBarrierError:
        _fail[0] = f"all_reduce token={tok} rank={r} BROKEN BARRIER (rank divergence)"
        raise
    with _lock:
        pieces = dict(_store[tok])
        del _store[tok]
    s = sum(pieces.values())
    t.copy_(s)


fake_dist = types.ModuleType("dist")
fake_dist.all_gather = _ag
fake_dist.broadcast = _bc
fake_dist.all_reduce = _ar
fake_dist.get_rank = lambda: _cur_rank[0]
fake_dist.get_world_size = lambda: WORLD
fake_dist.ReduceOp = types.SimpleNamespace(SUM="SUM")
fake_dist.init_process_group = lambda *a, **k: None
fake_dist.destroy_process_group = lambda: None
sys.modules["torch.distributed"] = fake_dist
torch.distributed = fake_dist
import torch.distributed as dist  # noqa: E402  (resolves to the fake)

# ---------------- shared geometry ----------------
from tensorlbm.octree_boundary.geometry import (  # noqa: E402
    build_octree_shell, sphere_distance_field,
)
from tensorlbm.amr_shell_planning import plan_body_shell_box  # noqa: E402
from tensorlbm.octree_boundary.l1_block import (  # noqa: E402
    L1BlockDistributed, gather_window_chunked, restrict_l1_block_to_coarse,
    step_l1_block_distributed, write_window_back,
)
from tensorlbm.octree_boundary.distributed_stepping import (  # noqa: E402
    interleaved_leaf_indices, step_octree_shell_distributed,
)
from tensorlbm.octree_boundary.stepping import _tau_chain  # noqa: E402
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, leaf_force_weights  # noqa: E402
from tensorlbm.cumulant import collide_cumulant_d3q27  # noqa: E402
from tensorlbm.d3q27 import equilibrium27, C as C27  # noqa: E402

nx, ny, nz = 96, 64, 64
radius = 6.0
center = (nx * 0.5, ny * 0.5, nz * 0.5)
bl = max(2.0, round(radius / 2.0))
dev = torch.device("cpu")
q = 27
u_in = 0.06

t0 = time.time()
solid_coarse = sphere_distance_field((nz, ny, nx), center, radius, dev) <= 0.0
plan = plan_body_shell_box(solid_coarse, shell_margin=6, wake_cells=32, pad=8)
box = plan.box
l1_shape = ((box.z1 - box.z0) * 2, (box.y1 - box.y0) * 2, (box.x1 - box.x0) * 2)
center_l1 = (center[0] * 2.0 - box.x0 * 2, center[1] * 2.0 - box.y0 * 2,
             center[2] * 2.0 - box.z0 * 2)
octree = build_octree_shell(
    l1_shape, center=center_l1, radius=radius * 2.0,
    bl_thickness_cells=bl, d_max=D_MAX, lattice="D3Q27", device=dev,
)
print(f"[sim] octree n_leaf={octree.n_leaf} built {time.time()-t0:.0f}s", flush=True)
n_leaf = octree.n_leaf
oshape = tuple(octree.meta["shape"])
tau_coarse = 0.5 + 3.0 * (u_in * radius / 100.0)
taus = _tau_chain(tau_coarse, octree.d_max)
tau_l1 = taus[1]
nx_local = nx // WORLD

S27 = [(int(C27[d, 0]), int(C27[d, 1]), int(C27[d, 2])) for d in range(27)]


def collide_coarse(f, tau):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


def stream27_roll(f):
    out = torch.empty_like(f)
    for d in range(27):
        sx, sy, sz = S27[d]
        out[d] = torch.roll(f[d], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


def advance_shell(f, tau, level, substep):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


def advance_l1(f, tau):
    f4 = f.view(q, 1, 1, -1)
    return collide_cumulant_d3q27(f4, tau, C_s=0.0).view_as(f)


# ---------------- per-rank driver ----------------
def run_rank(r, results):
    _cur_rank[0] = r
    lo = r * nx_local
    hi = lo + nx_local
    try:
        coarse_f = torch.zeros(q, nz, ny, nx_local + 2, device=dev)
        eq = equilibrium27(
            torch.ones(nz, ny, nx_local, device=dev),
            torch.full((nz, ny, nx_local), u_in, device=dev),
            torch.zeros(nz, ny, nx_local, device=dev),
            torch.zeros(nz, ny, nx_local, device=dev),
        )
        coarse_f[:, :, :, 1:-1] = eq
        coarse_f[:, :, :, 0:1] = eq[:, :, :, 0:1]
        coarse_f[:, :, :, -1:] = eq[:, :, :, -1:]

        if INTERLEAVE:
            lidx = interleaved_leaf_indices(n_leaf, WORLD, r)
        else:
            base, extra = divmod(n_leaf, WORLD)
            lo_l = base * r + min(r, extra)
            hi_l = lo_l + base + (1 if r < extra else 0)
            lidx = torch.arange(lo_l, hi_l, dtype=torch.int64)
        n_local = lidx.shape[0]

        octree_r = copy.copy(octree)
        eq_global = equilibrium27(
            torch.ones(oshape, device=dev),
            torch.full(oshape, u_in, device=dev),
            torch.zeros(oshape, device=dev),
            torch.zeros(oshape, device=dev),
        )
        octree_r.f_leaf = eq_global[:, octree.leaf_host_cell[lidx, 0],
                                    octree.leaf_host_cell[lidx, 1],
                                    octree.leaf_host_cell[lidx, 2]].clone()

        l1_block = L1BlockDistributed(
            box, (nz, ny, nx), tau_coarse, q=q, ratio=2, ghost=1,
            device=dev, solid_l1=octree._solid,
            collide_fn=advance_l1, stream_fn=stream27_roll,
        )
        l1_block.initialize_uniform(u_in)
        win = l1_block.win
        leaf_weights = leaf_force_weights(octree).to(dev)[lidx]

        def bfl_fn(octree_, out, post, gplan, ghost_vals, *, substep):
            return bfl_apply_gather(
                octree_, out, post, ghost_plan=gplan, ghost_vals=ghost_vals,
                force_weights=leaf_weights, return_force=True, q_min=None,
            )

        # ---- root step 1 ----
        coarse_old = coarse_f.clone()
        post = collide_coarse(coarse_f, tau_coarse)

        def halo_exchange(f_local):
            li = f_local[:, :, :, 1:2].contiguous()
            ri = f_local[:, :, :, -2:-1].contiguous()
            rg = [torch.empty_like(ri) for _ in range(WORLD)]
            dist.all_gather(rg, ri)
            f_local[:, :, :, 0:1] = rg[(r - 1) % WORLD]
            lg = [torch.empty_like(li) for _ in range(WORLD)]
            dist.all_gather(lg, li)
            f_local[:, :, :, -1:] = lg[(r + 1) % WORLD]

        halo_exchange(post)
        coarse_f = stream27_roll(post)

        cw_old, _ = gather_window_chunked(coarse_old, win, lo, hi,
                                          rank=r, world_size=WORLD)
        cw_new, in_slab = gather_window_chunked(coarse_f, win, lo, hi,
                                                rank=r, world_size=WORLD)
        cw_post, _ = gather_window_chunked(post, win, lo, hi,
                                           rank=r, world_size=WORLD)
        l1_phys_pre, l1_posts_phys, _pg = step_l1_block_distributed(
            l1_block, cw_old, cw_new)
        l1_f_phys = l1_block.physical_copy()
        ledger, mem, restricted, cells = step_octree_shell_distributed(
            octree_r, advance_shell, l1_phys_pre, l1_f_phys,
            tau_coarse=l1_block.tau_l1, l1_post=l1_posts_phys,
            ghost_plan=None, bfl_fn=bfl_fn, rank=r, world_size=WORLD,
            reflux=True, interleave=INTERLEAVE,
        )
        l1_block.set_physical(l1_f_phys)
        ledger2 = restrict_l1_block_to_coarse(l1_block, cw_new, cw_post)
        write_window_back(coarse_f, cw_new, win, in_slab, lo)
        fin = bool(torch.isfinite(octree_r.f_leaf).all())
        results[r] = ("OK" if fin else "NAN", float(octree_r.f_leaf.sum()),
                      float(ledger.mass_residual), float(ledger2.mass_residual))
        print(f"[r{r}] DONE finite={fin} f_sum={float(octree_r.f_leaf.sum()):.6g}", flush=True)
    except Exception as e:  # noqa: BLE001
        results[r] = ("ERR", repr(e))
        print(f"[r{r}] ERROR: {type(e).__name__}: {e}", flush=True)
        raise


results = {}
threads = [threading.Thread(target=run_rank, args=(r, results), daemon=True)
           for r in range(WORLD)]
t0 = time.time()
for th in threads:
    th.start()
alive = True
while any(th.is_alive() for th in threads):
    time.sleep(1)
    if time.time() - t0 > 900:
        print(f"[sim] TIMEOUT after 900s — threads still alive: "
              f"{[r for r, th in enumerate(threads) if th.is_alive()]}", flush=True)
        alive = False
        break
if alive:
    for th in threads:
        th.join(timeout=10)
print(f"[sim] total {time.time()-t0:.0f}s")
for r in range(WORLD):
    print(f"  rank{r}: {results.get(r)}")
