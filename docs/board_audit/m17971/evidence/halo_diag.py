import os, sys
sys.path.insert(0,'/root/tp-agent/work/m17868/xval/XVAL-LBM-LAX-PPERMUTE-PERIODIC-HALO')
import numpy as np, torch, torch.distributed as dist, arms
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world, init_method=os.environ["M"])
nz,ny,nx=16,8,8
s=arms.make_cpu_solver(nz,ny,nx,tau=0.6,halo_dtype=torch.float64)
f0=arms.make_field_3d(nz,ny,nx,0,kind="rand")
f=s.from_global(torch.as_tensor(f0))
s._halo_handles=s._start_halo_exchange(f); s._finalize_halo()
nzl=s.nz_local; z0=s.z_start_global
iu=list(s._idx_up); idn=list(s._idx_dn)
# expected left ghost = global plane (z0-1)%nz, idx_up ; right ghost = (z0+nzl)%nz, idx_dn
left_exp=torch.index_select(torch.as_tensor(f0)[:,(z0-1)%nz],0,s._idx_up)
right_exp=torch.index_select(torch.as_tensor(f0)[:,(z0+nzl)%nz],0,s._idx_dn)
dl=float((torch.index_select(f[:,0],0,s._idx_up)-left_exp).abs().max()); dr=float((torch.index_select(f[:,-1],0,s._idx_dn)-right_exp).abs().max())
print(f"rank {rank} z0={z0} left_ghost_err={dl:.3e} right_ghost_err={dr:.3e} halo_target_none={s._halo_target is None}", flush=True)
dist.destroy_process_group()
