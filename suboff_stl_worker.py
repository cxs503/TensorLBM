"""SUBOFF STL import test — uses read_stl + voxelize_stl + from_gradient."""
import sys, torch, numpy as np, functools, json, time
sys.path.insert(0, 'src')
from tensorlbm.stl_geometry import read_stl, voxelize_stl, SurfaceMesh_from_stl
from tensorlbm.drag_pressure import get_near_wall_3d, SurfaceMesh, drag_pressure_integration, drag_friction_integration
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.postprocess import detect_strouhal

device = torch.device(f"sdaa:{sys.argv[1] if len(sys.argv)>1 else 0}")
stl_path = sys.argv[2] if len(sys.argv)>2 else "/tmp/suboff_bare_hull_ascii.stl"
out_path = sys.argv[3] if len(sys.argv)>3 else "results_suboff_stl_test.json"

print(f"=== SUBOFF STL Re=1000 L=80 ===")
print(f"STL: {stl_path}")

# 1. Read STL
verts, faces, fnorms = read_stl(stl_path)
print(f"STL: {len(verts)} verts, {len(faces)} faces")

# 2. Scale to L=80 (STL has L=100)
scale = 80.0 / (verts[:,0].max() - verts[:,0].min())
verts_scaled = (verts * scale).astype(np.float32)
print(f"Scaled: L={verts_scaled[:,0].max()-verts_scaled[:,0].min():.1f}")

# 3. Voxelize
nx, ny, nz = 200, 80, 80
# Center the geometry
x_min = verts_scaled[:,0].min()
y_min = verts_scaled[:,1].min()
z_min = verts_scaled[:,2].min()
verts_centered = verts_scaled.copy()
verts_centered[:,0] -= x_min
verts_centered[:,1] -= y_min
verts_centered[:,2] -= z_min

solid_np = voxelize_stl(verts_centered, faces, (nz, ny, nx))
# Ensure shape is (nz, ny, nx) — voxelize_stl may return (nx, ny, nz)
if solid_np.shape != (nz, ny, nx):
    solid_np = solid_np.T if solid_np.ndim == 2 else np.transpose(solid_np, (2, 1, 0))
solid = torch.tensor(solid_np, dtype=torch.bool, device=device)
n_solid = solid.sum().item()
print(f"Grid: {nx}x{ny}x{nz}, solid={n_solid} ({100*n_solid/(nx*ny*nz):.1f}%)")

# 4. Near-wall + mesh (from_gradient — verified reliable)
near = get_near_wall_3d(solid)
n_near = near.sum().item()
print(f"Near-wall: {n_near}")

# Use from_gradient (verified more reliable than STL normals)
mesh = SurfaceMesh.from_gradient(solid, near)
print(f"Mesh: from_gradient")

# 5. Parameters
u_in = 0.06
nu = 0.0048
tau = 3*nu + 0.5
Cs = 0.05
# Estimate wetted area from near-wall cells
S_wet = float(n_near)
dpS = 0.5 * 1.0 * u_in**2 * S_wet
print(f"u_in={u_in}, tau={tau}, Cs={Cs}, dpS={dpS:.3f}, S_wet={S_wet:.0f}")

# 6. Init
f = equilibrium3d(
    torch.ones(nz, ny, nx, device=device),
    torch.zeros(nz, ny, nx, device=device),
    torch.zeros(nz, ny, nx, device=device),
    torch.zeros(nz, ny, nx, device=device)
).clone()

# 7. BC + collision
bc_config = {'u_in': u_in, 'direction': 'x', 'type': 'far_field'}
far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)
collide_fn = functools.partial(collide_smagorinsky_mrt3d, C_s=Cs)

# 8. Main loop
Cd_ref = 0.042
warmup = 1000
n_steps = 5000
cl_hist = []
t0 = time.time()

for step in range(1, n_steps+1):
    f = lbm_step_correct(f, collide_fn, tau, solid, u_in, far_field_fn)
    
    if step > warmup and step % 500 == 0:
        fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap='none', p0_method='near_wall')
        _, ffy, _ = drag_friction_integration(f, mesh, dpS, nu, formula='standard')
        Cd_p = fx_p / dpS
        Cd_f = ffy / dpS
        Cd_tot = Cd_p + Cd_f
        Cl = fy_p / dpS
        cl_hist.append(float(Cl))
        elapsed = time.time() - t0
        print(f"step={step}/{n_steps} Cd_p={Cd_p:.6f} Cd_f={Cd_f:.6f} Cd_tot={Cd_tot:.6f} (ref={Cd_ref}) Cl={Cl:.6f} ({elapsed:.0f}s)")
        sys.stdout.flush()

# 9. Final
fx_p, fy_p, _ = drag_pressure_integration(f, mesh, dpS, extrap='none', p0_method='near_wall')
_, ffy, _ = drag_friction_integration(f, mesh, dpS, nu, formula='standard')
Cd_p = fx_p / dpS
Cd_f = ffy / dpS
Cd_tot = Cd_p + Cd_f
err = abs(Cd_tot - Cd_ref) / Cd_ref * 100
St = detect_strouhal(cl_hist, 1.0, u_in, 80) if cl_hist else None
elapsed = time.time() - t0

print(f"=== FINAL === Cd_p={Cd_p:.6f} Cd_f={Cd_f:.6f} Cd_tot={Cd_tot:.6f} (ref={Cd_ref}) err={err:.1f}% St={St} time={elapsed:.0f}s")
result = {'Cd_p': float(Cd_p), 'Cd_f': float(Cd_f), 'Cd_tot': float(Cd_tot), 'err': float(err), 'St': St, 'method': 'STL+from_gradient'}
json.dump(result, open(out_path, 'w'))
print(f"Saved to {out_path}")
