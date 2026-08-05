"""Verify the pressure drag integration formula p*(sp-sm)*fluid against known analytical results.

The formula in wall_function_3d computes:
    drag_pres = Σ p * (sp - sm) * fluid
where:
    sp = torch.roll(solid, 1, dims=2)  # solid at x+1 neighbour
    sm = torch.roll(solid, -1, dims=2) # solid at x-1 neighbour
    
This computes p at the x-face of each solid cell:
    (sp=1,sm=0) → +p  (solid on right = front face, fluid pushes left → -x)
    (sp=0,sm=1) → -p  (solid on left = back face, fluid pushes right → +x)

Test cases with known analytical drag:
"""
import torch
import numpy as np


def pressure_drag_3d(p: torch.Tensor, solid: torch.Tensor) -> float:
    """Replicate the exact formula from wall_function_3d."""
    sp = torch.roll(solid, 1, dims=2)   # solid at +x neighbour
    sm = torch.roll(solid, -1, dims=2)  # solid at -x neighbour
    fluid = ~solid
    return float((p * (sp.to(p.dtype) - sm.to(p.dtype)) * fluid.to(p.dtype)).sum().item())


def analytical_drag_box(solid, p_left, p_right, p_top, p_bottom):
    """For a box solid [x0:x1,y0:y1,z0:z1], the net x-pressure force is
    (p_left - p_right) * (y1-y0) * (z1-z0)."""
    pass


def test():
    nx, ny, nz = 32, 16, 16
    
    print("=" * 60)
    print("Pressure Drag Integration Unit Tests")
    print("=" * 60)
    
    # ─── Test 1: Uniform pressure → zero net force ───
    print("\nTest 1: Uniform pressure p=1.0 → expected drag_pres=0")
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, :, 8:16] = True  # box from x=8 to x=15
    p = torch.ones(nz, ny, nx, dtype=torch.float32)
    dp = pressure_drag_3d(p, solid)
    print(f"  Result: {dp:.6f}  Expected: 0.0  {'✓' if abs(dp)<1e-6 else '✗ ERROR'}")
    
    # ─── Test 2: Box with p=10 front, p=0 back → 10*A ───
    print("\nTest 2: Box p=10 front, p=0 back, area=256 → expected drag=2560")
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, :, 10:20] = True  # box from 10 to 19, area=16*16=256
    p = torch.zeros(nz, ny, nx, dtype=torch.float32)
    p[:, :, :10] = 10.0   # front has p=10
    # Back has p=0 (default)
    dp = pressure_drag_3d(p, solid)
    expected = 10.0 * ny * nz
    print(f"  Result: {dp:.1f}  Expected: {expected:.1f}  {'✓' if abs(dp-expected)<1 else '✗ ERROR'}")
    
    # ─── Test 3: Box with p=0 front, p=10 back → -10*A (thrust) ───
    print("\nTest 3: Box p=0 front, p=10 back → expected drag=-2560")
    p = torch.zeros(nz, ny, nx, dtype=torch.float32)
    p[:, :, 20:] = 10.0   # back has p=10
    dp = pressure_drag_3d(p, solid)
    expected = -10.0 * ny * nz
    print(f"  Result: {dp:.1f}  Expected: {expected:.1f}  {'✓' if abs(dp-expected)<1 else '✗ ERROR'}")
    
    # ─── Test 4: SUBOFF-like pressure (front high, back low) → positive drag ───
    print("\nTest 4: Front p=1.01, back p=0.99, body in center → positive drag")
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, :, 12:20] = True  # box 12-19
    p = torch.ones(nz, ny, nx, dtype=torch.float32)
    # Higher pressure at front (x<12), lower at back (x>19)
    p[:, :, :12] = 1.01
    p[:, :, 20:] = 0.99
    dp = pressure_drag_3d(p, solid)
    area = ny * nz
    # Front face: p_front*area acts -x (positive drag), back face: p_back*area acts +x
    expected = (1.01 - 0.99) * area
    print(f"  Result: {dp:.3f}  Expected: {expected:.3f}  {'✓' if abs(dp-expected)<0.5 else '✗ ERROR'}")
    
    # ─── Test 5: Actual SUBOFF pressure field direction ───
    print("\nTest 5: Realistic SUBOFF pressure (stagnation front, suction back)")
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    # Simulate SUBOFF hull: long body, pressure drops from front to back
    cx = nx // 2
    # Simple hull shape
    for i in range(nz):
        for j in range(ny):
            solid[i, j, cx-8:cx+8] = True
    # Stagnation at front: p≈1.005, low pressure at mid-back: p≈0.997
    p = torch.ones(nz, ny, nx, dtype=torch.float32)
    for k in range(nx):
        if k < cx:
            p[:, :, k] = 1.0 + 0.005 * (1 - k/cx)  # decay from 1.005 to 1.0
        else:
            p[:, :, k] = 1.0 - 0.003 * ((k-cx)/(nx-cx))  # suction to 0.997
    dp = pressure_drag_3d(p, solid)
    
    # Manually compute: for each x-cross section, p * n_x * dA
    # Front face (at solid boundary, n_x = -1 → drag = positive)
    manual_front = 0
    manual_back = 0
    for k in range(1, nx):
        # Cells where solid[k-1]=False and solid[k]=True → front face
        front_mask = (~solid[:,:,k-1]) & solid[:,:,k]
        if front_mask.any():
            manual_front += float(p[:,:,k-1][front_mask].sum())
        # Cells where solid[k]=True and solid[k+1]=False → back face  
        back_mask = solid[:,:,k] & (~solid[:,:,min(k+1,nx-1)])
        if back_mask.any() and k < nx-1:
            manual_back += float(p[:,:,min(k+1,nx-1)][back_mask].sum())
    manual_drag = manual_front - manual_back
    print(f"  Formula drag:  {dp:.4f}")
    print(f"  Manual drag:   {manual_drag:.4f} (front={manual_front:.1f} - back={manual_back:.1f})")
    print(f"  {'✓ Match' if abs(dp-manual_drag)<1.0 else '✗ MISMATCH'}")
    
    # ─── Test 6: Sign convention check ───
    print("\nTest 6: Sign convention — which direction does positive drag represent?")
    print("  If p*(sp-sm)*fluid gives positive value when front pressure > back pressure,")
    print("  the sign convention is: positive drag = force in +x direction (drag opposes flow)")
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[:, :, 12:20] = True
    p = torch.ones(nz, ny, nx, dtype=torch.float32)
    p[:, :, :12] = 1.01   # high pressure in front of body
    p[:, :, 20:] = 0.99   # low pressure behind body
    dp = pressure_drag_3d(p, solid)
    print(f"  Front high(1.01), back low(0.99) → drag = {dp:.3f}")
    if dp > 0:
        print("  ✓ Positive drag = force in +x (correct: net force pushes body downstream)")
    else:
        print("  ✗ Negative drag = force in -x (WRONG: pressure difference should push +x)")

if __name__ == "__main__":
    test()
