"""Conv-FDM prototype — stencil-as-convolution finite difference.

Each FD stencil becomes a Conv3D kernel:
  ∂/∂x  = [-1, 0, 1]/(2dx)  via conv3d with (1,1,1,1,3) kernel
  ∇²    = [1, -2, 1]/dx²    via conv3d with (1,1,1,1,3) kernel
  u·∇   = u_x*∂/∂x + u_y*∂/∂y + u_z*∂/∂z  via element-wise + conv

Memory: 4 fields (u,v,w,p) vs 19 distributions → 4.75× less memory.
Target: 500-2000 MLUPS on SDAA vs 103 MLUPS for LBM.
"""
import torch
import torch.nn.functional as F
import math, time


class ConvFDMStep:
    """Single NS time step via Conv3D stencils."""

    def __init__(self, dx=1.0, dt=1.0, nu=0.01, device='cpu'):
        self.dx = dx; self.dt = dt; self.nu = nu
        dev = torch.device(device)

        # Central difference stencil: [-1, 0, 1]/(2dx)
        # Conv3D expects (out_ch, in_ch, kd, kh, kw)
        c = 1.0 / (2 * dx)
        self.ddx = torch.tensor([[-c, 0, c]], device=dev).reshape(1,1,1,1,3)
        self.ddy = torch.tensor([[[-c],[0],[c]]], device=dev).reshape(1,1,1,3,1)
        self.ddz = torch.tensor([[[[-c]],[[0]],[[c]]]], device=dev).reshape(1,1,3,1,1)

        # Laplacian: [1, -2, 1]/dx²
        l = 1.0 / (dx * dx)
        self.lapx = torch.tensor([[l, -2*l, l]], device=dev).reshape(1,1,1,1,3)
        self.lapy = torch.tensor([[[l],[-2*l],[l]]], device=dev).reshape(1,1,1,3,1)
        self.lapz = torch.tensor([[[[l]],[[-2*l]],[[l]]]], device=dev).reshape(1,1,3,1,1)

        # For pressure Poisson: repeated Jacobi smoothing
        self.lap_sum = self.lapx + self.lapy + self.lapz
        self._diag = -4.0 * l  # diagonal of 2D Laplacian (for CFD we use 3D: -6l)

    def grad(self, f4):
        """Return (df/dx, df/dy, df/dz) via Conv3D with padding."""
        p = (0,0,0,0,1)
        return (F.conv3d(f4, self.ddx, padding=(0,0,1)),
                F.conv3d(f4, self.ddy, padding=(0,1,0)),
                F.conv3d(f4, self.ddz, padding=(1,0,0)))

    def laplacian(self, f4):
        """∇²f via Conv3D with padding."""
        return (F.conv3d(f4, self.lapx, padding=(0,0,1)) +
                F.conv3d(f4, self.lapy, padding=(0,1,0)) +
                F.conv3d(f4, self.lapz, padding=(1,0,0)))

    def advection(self, u4, v4, w4):
        """(u·∇)u term: u*du/dx + v*du/dy + w*du/dz etc."""
        dxu, dyu, dzu = self.grad(u4)
        dxv, dyv, dzv = self.grad(v4)
        dxw, dyw, dzw = self.grad(w4)
        adv_u = u4 * dxu + v4 * dyu + w4 * dzu
        adv_v = u4 * dxv + v4 * dyv + w4 * dzv
        adv_w = u4 * dxw + v4 * dyw + w4 * dzw
        return adv_u, adv_v, adv_w

    def pressure_poisson(self, div, n_iter=20):
        """Solve ∇²p = -div using Jacobi iteration."""
        rhs = -div / self.dt
        p = torch.zeros_like(rhs)
        diag_inv = -1.0 / (6.0 * self.lapx[0,0,0,0,1].abs().item())
        for _ in range(n_iter):
            lap_p = self.laplacian(p)
            p = p + diag_inv * (lap_p - rhs)
        return p

    def step(self, u, v, w, solid=None):
        """One explicit time step of Navier-Stokes.

        Args:
            u, v, w: velocity fields (1,1,nz,ny,nx).
            solid: optional mask with 0=fluid, 1=solid.
        """
        u4 = u.unsqueeze(0).unsqueeze(0) if u.ndim == 3 else u.unsqueeze(0)
        v4 = v.unsqueeze(0).unsqueeze(0) if v.ndim == 3 else v.unsqueeze(0)
        w4 = w.unsqueeze(0).unsqueeze(0) if w.ndim == 3 else w.unsqueeze(0)

        # Advection + diffusion
        au, av, aw = self.advection(u4, v4, w4)
        lu, lv, lw = self.laplacian(u4), self.laplacian(v4), self.laplacian(w4)

        # Intermediate velocity (explicit Euler, no pressure)
        us = u4 + self.dt * (-au + self.nu * lu)
        vs = v4 + self.dt * (-av + self.nu * lv)
        ws = w4 + self.dt * (-aw + self.nu * lw)

        # Divergence of intermediate velocity
        du, _, _ = self.grad(us)
        _, dv, _ = self.grad(vs)
        _, _, dw = self.grad(ws)
        div = du + dv + dw

        # Pressure correction
        p = self.pressure_poisson(div)
        px, py, pz = self.grad(p)

        un = (us - self.dt * px).squeeze(0)
        vn = (vs - self.dt * py).squeeze(0)
        wn = (ws - self.dt * pz).squeeze(0)

        # Solid mask: zero velocity at wall
        if solid is not None:
            un = un * (~solid).float()
            vn = vn * (~solid).float()
            wn = wn * (~solid).float()

        return un.squeeze(0), vn.squeeze(0), wn.squeeze(0)


def benchmark(FDMClass, device, nx, ny, nz, nu, n_steps=50):
    """Benchmark Conv-FDM speed on given grid."""
    dev = torch.device(device)
    fdm = FDMClass(nu=nu, device=device)
    u = torch.randn(1, 1, nz, ny, nx, device=dev) * 0.01
    v = torch.randn(1, 1, nz, ny, nx, device=dev) * 0.01
    w = torch.randn(1, 1, nz, ny, nx, device=dev) * 0.01
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=dev) if n_steps > 0 else None

    # Warmup
    fdm.step(u.squeeze(), v.squeeze(), w.squeeze(), solid)
    if device.startswith('sdaa'):
        torch.sdaa.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_steps):
        u, v, w = fdm.step(u.squeeze(), v.squeeze(), w.squeeze(), solid)
    if device.startswith('sdaa'):
        torch.sdaa.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()
    ms = (time.time() - t0) / n_steps * 1000
    cells = nx * ny * nz
    mlup = 4 * cells / (ms / 1000) / 1e6  # 4 fields updated per cell
    return ms, mlup


if __name__ == '__main__':
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else 'sdaa:0'
    for nx in [160, 256, 320]:
        ny = nz = int(nx * 0.4)
        ms, mlup = benchmark(ConvFDMStep, dev, nx, ny, nz, nu=0.01, n_steps=20)
        print(f'{nx}x{ny}x{nz} {nx*ny*nz/1e6:.1f}M cells: {ms:.0f}ms/step  {mlup:.0f} MLUP')
        print(f'  vs LBM 370ms @ 1.3M → estimated FDM: {ms:.0f}ms')
