"""验证STL法向是否正确指向流体格点。
在仿真前校对: 对每个近壁流体格点, 检查法向是否指向流体侧。
"""
import sys, numpy as np, torch
sys.path.insert(0, 'src')
from tensorlbm.stl_geometry import read_stl, voxelize_stl, SurfaceMesh_from_stl
from tensorlbm.drag_pressure import get_near_wall_3d

stl_path = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/ship-performance-platform-incoming/ship-performance-platform/backend/data/geometry/ships/KVLCC2_Hull.stl"

# 1. 读STL
verts, faces, fnorms = read_stl(stl_path)
print(f"STL: {len(faces)} faces, {len(verts)} vertices")

# 2. 体素化 (小网格快速验证)
nx, ny, nz = 100, 40, 40
solid = voxelize_stl(verts, faces, (nz, ny, nx))
solid = torch.tensor(solid, dtype=torch.bool)
n_solid = solid.sum().item()
print(f"Grid: {nz}x{ny}x{nx}, solid={n_solid} ({100*n_solid/(nx*ny*nz):.1f}%)")

# 3. 近壁mask
near = get_near_wall_3d(solid)
n_near = near.sum().item()
print(f"Near-wall: {n_near}")

# 4. 流体近壁格点
near_idx = near.nonzero(as_tuple=False)
solid_np = solid.numpy()
is_fluid = ~solid_np[near_idx[:,0], near_idx[:,1], near_idx[:,2]]
n_fluid_near = is_fluid.sum()
n_solid_near = (~is_fluid).sum()
print(f"  Fluid near-wall: {n_fluid_near} ({100*n_fluid_near/n_near:.1f}%)")
print(f"  Solid near-wall: {n_solid_near} ({100*n_solid_near/n_near:.1f}%)")

# 5. 构建mesh (用修复后的代码)
origin = (0.0, 0.0, 0.0)
spacing = (1.0, 1.0, 1.0)
mesh = SurfaceMesh_from_stl(solid, near, verts, faces, fnorms, origin, spacing)

# 6. 验证法向方向
# 对每个流体近壁格点, 检查法向是否指向流体(即法向方向上有流体格点)
nx_n = mesh.nx_n.numpy()
ny_n = mesh.ny_n.numpy()
nz_n = mesh.nz_n.numpy()

fluid_idx = near_idx[is_fluid]
correct = 0
wrong = 0
for idx in fluid_idx:
    iz, iy, ix = idx[0].item(), idx[1].item(), idx[2].item()
    nx_v, ny_v, nz_v = nx_n[iz, iy, ix], ny_n[iz, iy, ix], nz_n[iz, iy, ix]
    # 法向方向上走1步, 检查是否是流体
    dx = int(np.sign(nx_v))
    dy = int(np.sign(ny_v))
    dz = int(np.sign(nz_v))
    niz = min(max(iz + dz, 0), nz-1)
    niy = min(max(iy + dy, 0), ny-1)
    nix = min(max(ix + dx, 0), nx-1)
    if not solid_np[niz, niy, nix]:
        correct += 1
    else:
        wrong += 1

total = correct + wrong
if total > 0:
    print(f"\n=== 法向校对结果 ===")
    print(f"正确(指向流体): {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"错误(指向固体): {wrong}/{total} ({100*wrong/total:.1f}%)")
    
    # 压力代理: 均匀p=1, 检查Cd_p符号
    p_proxy = np.ones_like(solid_np, dtype=np.float64)
    mask = (~solid_np) & near.numpy()
    cd_p_x = (p_proxy * nx_n * mask).sum()
    cd_p_y = (p_proxy * ny_n * mask).sum()
    cd_p_z = (p_proxy * nz_n * mask).sum()
    print(f"\n压力代理(均匀p=1):")
    print(f"  Cd_p_x = {cd_p_x:+.4f} (应正, x方向阻力)")
    print(f"  Cd_p_y = {cd_p_y:+.4f}")
    print(f"  Cd_p_z = {cd_p_z:+.4f}")
    
    if cd_p_x > 0:
        print(f"\n✓ 法向正确! Cd_p_x > 0")
    else:
        print(f"\n✗ 法向错误! Cd_p_x < 0")
else:
    print("无流体近壁格点!")
