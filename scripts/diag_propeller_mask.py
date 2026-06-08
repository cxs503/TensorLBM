#!/usr/bin/env python3
"""Diagnostic: propeller mask quality check."""
import torch
from tensorlbm.propeller_cad import PropellerGeometryConfig, build_propeller_mask

geo = PropellerGeometryConfig(
    n_blades=5, diameter=48.0, hub_diameter_ratio=0.18,
    hub_length_ratio=0.45, pitch_ratio_07=0.95,
    blade_area_ratio=0.65, skew_deg=0.0, rake_ratio=0.0,
    max_thickness_ratio=0.06,
)

R = geo.radius
R_hub = geo.hub_radius
print(f"Radius={R}, Hub_radius={R_hub:.2f}, Hub_len={geo.hub_length:.1f}")
print(f"Mean chord: {geo.mean_chord:.2f} lu")
print(f"Hub volume (ideal): {3.14159 * R_hub**2 * geo.hub_length:.0f} cells")
blade_area = geo.blade_area_ratio * 3.14159 * R**2
print(f"Blade projected area: {blade_area:.0f} lu^2")
expected = int(3.14159 * R_hub**2 * geo.hub_length + blade_area * 5 * 0.1)
print(f"Expected solid >= {expected}")

for nx, ny, nz in [(120, 60, 60), (160, 80, 80), (200, 100, 100)]:
    cx, cy, cz = int(nx * 0.35), ny // 2, nz // 2
    mask = build_propeller_mask(nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz, config=geo)
    solid = int(mask.sum().item())
    total = nx * ny * nz
    print(f"  Grid {nx}x{ny}x{nz}: {solid}/{total} ({100*solid/total:.1f}%)")

# Radial distribution
mask2 = build_propeller_mask(nx=200, ny=100, nz=100, cx=70, cy=50, cz=50, config=geo)
yy, zz, xx = torch.meshgrid(
    torch.arange(100, dtype=torch.float32),
    torch.arange(100, dtype=torch.float32),
    torch.arange(200, dtype=torch.float32),
    indexing="ij",
)
r = torch.sqrt((yy - 50) ** 2 + (zz - 50) ** 2)
for r_min, r_max in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 24)]:
    in_bin = (r >= r_min) & (r < r_max)
    count = int((mask2 & in_bin).sum().item())
    print(f"  r=[{r_min},{r_max}): {count}")
