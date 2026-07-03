"""Ahmed body benchmark — automotive drag reference.

Standard automotive CFD benchmark: simplified car body with slant back.
Experimental Cd (Ahmed 1984):
  25° slant: Cd ≈ 0.28 (low drag)
  35° slant: Cd ≈ 0.43 (high drag, flow separation)

Geometry: L=1.044m, W=0.388m, H=0.288m, slant angle 25° or 35°
Re = 4.3e6 (based on length)
"""
from __future__ import annotations
import sys, math, torch
sys.path.insert(0, 'src')
from tensorlbm.cp_measurement import print_cp_report
from tensorlbm.d3q27 import equilibrium27, macroscopic27, C as C27, correct_mass27
from tensorlbm.cumulant import collide_cumulant_d3q27

KAPPA=0.41; B_CONST=5.0
SHIFTS=[(int(C27[q,0]),int(C27[q,1]),int(C27[q,2])) for q in range(27)]
def stream27(f):
    out=torch.empty_like(f)
    for q in range(27):
        sx,sy,sz=SHIFTS[q]
        out[q]=torch.roll(f[q],shifts=(sz,sy,sx),dims=(0,1,2))
    return out
def far_field(f,u):
    nz,ny,nx=f.shape[1],f.shape[2],f.shape[3]
    r=torch.ones(nz,ny,nx,dtype=f.dtype,device=f.device)
    feq=equilibrium27(r,torch.full_like(r,u),torch.zeros_like(r),torch.zeros_like(r))
    f=f.clone()
    f[:,:,:,0]=feq[:,:,:,0];f[:,:,:,-1]=f[:,:,:,-2]
    f[:,0,:,:]=feq[:,0,:,:];f[:,-1,:,:]=feq[:,-1,:,:]
    f[:,:,0,:]=feq[:,:,0,:];f[:,:,-1,:]=feq[:,:,-1,:]
    return f

def build_ahmed_body(nx,ny,nz,slant_deg=25.0,device='cpu'):
    """Build Ahmed body mask. Body centered in domain."""
    L=int(nx*0.35); W=int(ny*0.4); H=int(nz*0.35)
    cx,cy,cz=nx*0.35,ny/2,nz*0.35
    slant_len=int(L*0.3)  # slant portion
    body_len=L-slant_len
    zz,yy,xx=torch.meshgrid(torch.arange(nz),torch.arange(ny),torch.arange(nx),indexing='ij')
    x0=cx; x1=cx+body_len; x2=cx+L
    # Main body (rectangular)
    in_body=(xx>=x0)&(xx<x1)&(yy>=cy-W/2)&(yy<cy+W/2)&(zz>=cz)&(zz<cz+H)
    # Slant back (angled)
    slant_h=H*0.6  # slant drops to 60% height
    slant_t=(xx-x1).clamp(min=0)/max(slant_len,1)  # 0 to 1 along slant
    slant_z=cz+H-slant_t*(H-slant_h)
    in_slant=(xx>=x1)&(xx<x2)&(yy>=cy-W/2)&(yy<cy+W/2)&(zz>=cz)&(zz<slant_z)
    # Round front
    front_r=W*0.3
    dx=(xx-x0).clamp(min=0); dy=torch.minimum((yy-(cy-W/2)).clamp(min=0),(cy+W/2-yy).clamp(min=0))
    in_front=(xx<x0)&(dx**2+dy**2<front_r**2)&(zz>=cz)&(zz<cz+H)
    solid=in_body|in_slant|in_front
    return solid.to(device)

def run_ahmed(slant_deg=25.0, device='sdaa:0', n_steps=1500, warmup=400):
    dev=torch.device(device)
    nx,ny,nz=320,128,96; u_in=0.06; re=1e6
    nu_lat=u_in*(nx*0.35)/re; tau=3.0*nu_lat+0.5
    solid=build_ahmed_body(nx,ny,nz,slant_deg,device='cpu').to(dev)
    fluid=~solid
    W=ny*0.4; H=nz*0.35; S=W*H  # frontal area for bluff body
    dyn_p_S=0.5*1.0*u_in**2*S
    # Near-wall mask
    nbrs=torch.zeros_like(solid)
    for ax,sgn in [(2,1),(2,-1),(1,1),(1,-1),(0,1),(0,-1)]:
        nbrs|=(solid&torch.roll(fluid,sgn,dims=ax))
    near=nbrs
    c=C27.to(dev).float()
    cx=c[:,0].view(27,1,1,1);cy=c[:,1].view(27,1,1,1);cz=c[:,2].view(27,1,1,1)
    w27=torch.tensor([8/27]+[2/27]*6+[1/54]*12+[1/216]*8,dtype=torch.float32,device=dev).view(27,1,1,1)
    cs2=1.0/3.0
    rho0=torch.ones(nz,ny,nx,device=dev)
    ux0=torch.full((nz,ny,nx),u_in,device=dev);ux0[solid]=0
    f=equilibrium27(rho0,ux0,torch.zeros_like(ux0),torch.zeros_like(ux0))
    im=float(rho0.sum().item())
    cd_ref=0.28 if slant_deg<30 else 0.43
    print(f'Ahmed body {slant_deg}°: Re={re:.0e} grid={nx}x{ny}x{nz} Cd_ref={cd_ref}',flush=True)
    fric=[];pres=[];import time;t0=time.time()
    for step in range(1,n_steps+1):
        f=collide_cumulant_d3q27(f,tau=tau);f=stream27(f)
        rho,ux,uy,uz=macroscopic27(f)
        u_mag=torch.sqrt(ux*ux+uy*uy+uz*uz).clamp(min=1e-12)
        u_tau=torch.sqrt(nu_lat*u_mag/0.5).clamp(min=1e-12)
        y_plus=0.5*u_tau/nu_lat;turb=(y_plus>11.6)&near
        if bool(turb.any()):
            ut=u_tau[turb].clone();um=u_mag[turb]
            for _ in range(8):
                lyp=torch.log(0.5*ut/nu_lat);fv=ut*(lyp/KAPPA+B_CONST)-um
                fp=(lyp/KAPPA+B_CONST)+1.0/KAPPA;ut=(ut-fv/fp.clamp(min=1e-10)).clamp(min=1e-12)
            u_tau[turb]=ut
        tau_w=u_tau*u_tau;inv_umag=1.0/u_mag;coef=-(tau_w/0.5)*near.to(f.dtype)
        fx=coef*(ux*inv_umag);fy=coef*(uy*inv_umag);fz=coef*(uz*inv_umag)
        cu=cx*ux+cy*uy+cz*uz;forcing=w27*(1.0+cu/cs2)*(cx*fx+cy*fy+cz*fz)/cs2;f=f+forcing
        df=(tau_w*(ux*inv_umag)*near.to(f.dtype)).sum().item()
        p=(rho-1.0)/3.0;sp=torch.roll(solid,1,dims=2);sm=torch.roll(solid,-1,dims=2)
        dp=(p*(sp.to(f.dtype)-sm.to(f.dtype))*fluid.to(f.dtype)).sum().item()
        f=far_field(f,u_in)
        if step%100==0:f=correct_mass27(f,im)
        if step>warmup and math.isfinite(df):fric.append(df);pres.append(dp)
        if step%300==0 or step==n_steps:
            cf=sum(fric)/max(len(fric),1)/dyn_p_S;cp=sum(pres)/max(len(pres),1)/dyn_p_S
            print(f'  step {step}: Cf={cf:.4f} Cp={cp:.4f} Cd={cf+cp:.4f} (ref={cd_ref})',flush=True)
    dt=time.time()-t0
    cf=sum(fric)/max(len(fric),1)/dyn_p_S;cp=sum(pres)/max(len(pres),1)/dyn_p_S
    print_cp_report(f, solid, u_in, geometry_type='building', cx=cx, cy=cy)
    print(f'Final: Cd={cf+cp:.4f} ref={cd_ref} {dt:.0f}s',flush=True)
    return cf+cp

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser()
    a.add_argument('--device',default='sdaa:0')
    a.add_argument('--slant',type=float,default=25.0)
    args=a.parse_args()
    run_ahmed(slant_deg=args.slant,device=args.device)
