"""Cycle-resolved D2Q5 diffusion with transverse wall heating and periodic y."""
import json
import math

import torch

from tensorlbm.scalar import (
    scalar_collide_bgk, scalar_diffusivity, scalar_equilibrium, scalar_link_transfer,
    scalar_stream_slab, tau_from_diffusivity,
)


def measure(n, insulated):
    dx, period = 1/n, .08
    steps = math.ceil(period/(scalar_diffusivity(.9)*dx**2))
    dt = period/steps
    tau = tau_from_diffusivity(dt/dx**2)
    omega, b = 2*math.pi/period, 2*math.pi
    wave = complex(b*b, omega)**.5
    faces = torch.arange(n+1, dtype=torch.float64)*dx
    centers = .5*(faces[:-1]+faces[1:])
    y_mode = torch.cos(b*centers)*math.sin(b*dx/2)/(b*dx/2)
    # Cell-average temperature, x-face flux averaged along y, and y-face
    # flux averaged along x, for unit conductivity and volumetric capacity.
    z = wave*(1-faces)
    if insulated:
        denom = torch.cosh(z[0])
        x_mode = -torch.sinh(z).diff()/(dx*wave*denom)
        qx = wave*torch.sinh(z)/denom
    else:
        denom = torch.sinh(z[0])
        x_mode = -torch.cosh(z).diff()/(dx*wave*denom)
        qx = wave*torch.cosh(z)/denom
    exact = y_mode[:,None]*x_mode
    ref_x = y_mode[:,None]*qx
    ref_y = b*torch.sin(b*faces[1:,None])*x_mode
    field = exact.real.clone()
    zero = torch.zeros_like(field)
    g = scalar_equilibrium(field,zero,zero)
    temp = torch.zeros_like(exact)
    power_x = torch.zeros_like(ref_x)
    power_y = torch.zeros_like(ref_y)
    max_residual = 0.
    for i in range(steps):
        post = scalar_collide_bgk(g,field,zero,zero,tau)
        angle = omega*(i+.5)*dt
        g, walls = scalar_stream_slab(post, y_mode*math.cos(angle), None if insulated else 0.)
        tx = torch.cat((walls[0,:,None], scalar_link_transfer(post,"x"), -walls[1,:,None]), dim=1)
        ty = scalar_link_transfer(post,"y",periodic=True)
        next_field = g.sum(0)
        residual = next_field-field+tx.diff(dim=1)+ty-torch.roll(ty,1,0)
        max_residual = max(max_residual,float(residual.abs().max())*dx**2)
        phase = complex(math.cos(angle), -math.sin(angle))
        power_x += phase*tx
        power_y += phase*ty
        # Trapezoidal projection at temperature sampling times.
        temp += (complex(math.cos(omega*i*dt),-math.sin(omega*i*dt))*field
                 + complex(math.cos(omega*(i+1)*dt),-math.sin(omega*(i+1)*dt))*next_field)/steps
        field = next_field
    scale = 2*dx/(steps*dt)
    return dict(n=n, east_wall="insulated" if insulated else "fixed", steps=steps,
                temperature_error_K=float((temp-exact).abs().max()),
                x_flux_error_W_m2=float((power_x*scale-ref_x).abs().max()),
                y_flux_error_W_m2=float((power_y*scale-ref_y).abs().max()),
                cell_balance_J=float(max_residual))


def main():
    rows = []
    for insulated in (False,True):
        previous = None
        for n in (8,16,32,64):
            row = measure(n,insulated)
            errors = [row[k] for k in ("temperature_error_K","x_flux_error_W_m2","y_flux_error_W_m2")]
            passed = all(math.isfinite(e) for e in errors) and row["cell_balance_J"] < 1e-14
            if previous:
                row["error_reduction"] = [old/e for old,e in zip(previous,errors)]
                passed &= all(ratio > 3 for ratio in row["error_reduction"])
            if n == 64:
                passed &= all(e < limit for e,limit in zip(errors,[.001,.015,.01]))
            row["passed"] = passed
            previous = errors
            rows.append(row)
    print(json.dumps(rows,indent=2))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
