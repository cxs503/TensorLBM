"""Per-rank solid/wall-load estimate for SUBOFF x-slab decomposition (theoretical grounding)."""
import sys
sys.path.insert(0, 'src')
import torch
from tensorlbm.suboff_cad import build_suboff_mask

nx, ny, nz = 384, 160, 160
hull_length = 206.0          # task: hull ~206 cells
world = 16
nx_local = nx // world       # 24 columns/rank
cx = 0.35 * nx               # production placement

for htype in ('bare_hull', 'full'):
    solid, stats = build_suboff_mask(
        hull_type=htype, nx=nx, ny=ny, nz=nz,
        cx=cx, cy=ny/2.0, cz=nz/2.0, length=hull_length, device='cpu')
    solid = solid.bool()  # (nz, ny, nx)
    fluid = ~solid

    # wall links (BB useful work): fluid cell with >=1 solid neighbor per axis direction
    nbrs = torch.zeros_like(solid)
    for ax, sgn in [(0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)]:
        nbrs |= (torch.roll(solid, sgn, dims=ax) & fluid)
    wall_links = (nbrs & fluid).sum().item()

    x_lo = int((solid.any(dim=(0,1))).nonzero()[0].item())
    x_hi = int((solid.any(dim=(0,1))).nonzero()[-1].item())

    print(f"=== hull_type={htype} hull={hull_length:.0f} grid={nx}x{ny}x{nz} ===")
    print(f"total solid cells = {int(solid.sum())}, wall-link cells = {int(wall_links)}")
    print(f"hull x-span = [{x_lo}, {x_hi}]  (hull occupies ranks {x_lo//nx_local}..{x_hi//nx_local})")
    print(f"{'rank':>4} {'x-range':>12} {'solid':>8} {'solid%':>7} {'wall':>7} {'wall%':>7}")
    for r in range(world):
        s = x_lo + r * nx_local
        seg = solid[:, :, s:s+nx_local]
        wall = ((nbrs & fluid)[:, :, s:s+nx_local]).sum().item()
        sc = int(seg.sum())
        tot = int(solid.sum())
        wtot = int(wall_links)
        print(f"{r:>4} {f'[{s},{s+nx_local})':>12} {sc:>8} {100.0*sc/tot:>6.1f}% {wall:>7} {100.0*wall/wtot:>6.1f}%")
    # max/min solid ratio
    per_rank = [int(solid[:, :, r*nx_local:(r+1)*nx_local].sum()) for r in range(world)]
    print(f"solid per rank: min={min(per_rank)} max={max(per_rank)} (ratio {max(per_rank)/max(min(per_rank),1):.1f}x over non-empty ranks)")
    print()
