"""Smoke: curvature estimator + tapered-channel ladder (short)."""
import math, sys, time
import torch
torch.set_num_threads(1)
from tensorlbm import cac_lbm2d as cac
from tensorlbm import pc_ladder as P

CFG = dict(rho_l=1.0, rho_g=0.1, mu_l=0.2, mu_g=0.02, sigma=0.05, width=3.0,
           mobility=0.05, mu_law="harmonic", p0=0.01, gravity=(0.0, 0.0),
           gravity_mode="rho", axes="pp")


def droplet_case(steps=600):
    shape = (64, 64)
    R0 = 20.0
    sim = cac.CAC2D(shape, **CFG)
    sim.initialize(cac.init_phi_droplet(shape, R0, CFG["width"], center=(31.5, 31.5)))
    t0 = time.time()
    cac.run(sim, steps, sample_every=0)
    m = cac.macro(sim)
    phi, p = m["phi"], m["p"]
    kap = P.curvature_field(phi)
    band = (phi > 0.3) & (phi < 0.7)
    kmean = float(kap[band].mean())
    # apparent radius from phi=0.5 contour: liquid area
    area = float((phi > 0.5).sum().item())
    Req = math.sqrt(area / math.pi)
    # pressure difference: liquid core vs far gas
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    rr = ((yy - 31.5) ** 2 + (xx - 31.5) ** 2).sqrt()
    pl = float(p[rr < Req - 4].mean())
    pg = float(p[rr > Req + 8].mean())
    print(f"[droplet] steps={steps} kappa_band={kmean:.5f} 1/Req={1/Req:.5f} "
          f"1/R0={1/R0:.5f} Req={Req:.2f} dp={pl-pg:.5e} sigma*kappa={0.05*kmean:.5e} "
          f"dp/(kappa)={((pl-pg)/kmean if kmean else float('nan')):.4f} ({time.time()-t0:.0f}s)",
          flush=True)


def tapered_case(g, steps=800):
    ny, nx = 44, 128
    shape = (ny, nx)
    mask, r_rows = P.tapered_channel(ny, nx, wall_t=2, r_wide=20.0, r_min=8.0,
                                     throat_center=78.0, throat_half_length=16.0)
    sim = cac.CAC2D(shape, wall_mask=mask, wall_phi_policy="contact",
                    contact_angle_deg=60.0, **CFG)
    sim.initialize(P.plug_phi(shape, width=CFG["width"], x_lo=26.0, x_hi=56.0))
    sim.gravity = (g, 0.0)
    t0 = time.time()
    for k in range(steps // 100):
        cac.run(sim, 100, sample_every=0)
        st = P.meniscus_state(sim.f.sum(dim=0), fluid_mask=sim.fluid_mask)
        d = cac.diagnostics(sim)
        print(f"  step={sim.step_count} x_lead={st['x_lead']:.0f} x_trail={st['x_trail']:.0f} "
              f"Lp={st['plug_length']:.0f} kap=({st['kappa_trail']:.4f},{st['kappa_lead']:.4f}) "
              f"umax={d['u_max_fluid']:.2e} nan={d['nan_f']}", flush=True)
        if d["nan_f"]:
            break
    print(f"[taper] g={g:.3e} wall={time.time()-t0:.0f}s R_wide_rows={int(r_rows.max())} "
          f"R_min_rows={int(r_rows.min())}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "drop"
    if what == "drop":
        droplet_case()
    else:
        tapered_case(float(sys.argv[2]))