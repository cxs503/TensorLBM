from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

out=Path('docs/assets/ice-coupling')
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for name in ('dry','wet','time_half','exchange_half','particle_fine','intact_control'):
    state=json.loads((out/(name+'.json')).read_text())['state'];h=state['history']
    t=[r['time_s'] for r in h]
    axes[0,0].plot(t,[r['total_fy_n'] for r in h],label=name)
    axes[0,1].plot(t,[r['broken_bonds'] for r in h],label=name)
    axes[1,0].plot(t,[r['dem_discrete_energy_residual_J'] for r in h],label=name)
    axes[1,1].plot(t,[r['staggered_interface_work_residual_J'] for r in h],label=name)
for ax,title,unit in zip(axes.flat,['Tool force','Broken bonds (resolution dependent)','DEM discrete energy residual','Staggered interface work residual'],['N','count','J','J']):
    ax.set_title(title);ax.set_xlabel('Physical time [s]');ax.set_ylabel(unit);ax.grid(alpha=.3)
axes[0,0].legend(fontsize=8)
fig.savefig(out/'histories.png',dpi=180)
plt.close(fig)
state=json.loads((out/'wet.json').read_text())['state'];dem=state['dem']
f=np.array(state['f']);C=np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
rho=f.sum(0);u=np.einsum('qyx,qd->yxd',f,C)/rho[:,:,None]*.025/.0005
ny,nx=rho.shape;x=np.arange(nx)*.025-nx*.025/2;y=np.arange(ny)*.025-ny*.025/2
fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
mesh=ax.pcolormesh(x,y,np.linalg.norm(u,axis=-1),shading='nearest',cmap='Blues')
fig.colorbar(mesh,ax=ax,label='Liquid speed [m/s]')
p=np.array(dem['positions_m']);initial=np.array(dem['initial_positions_m'])
# Rebuild deterministic square-lattice pair IDs (DEM uses full upper triangle).
i,j=np.triu_indices(len(p),1);alive=np.array(dem['alive'])
segments=np.stack((p[i[alive]],p[j[alive]]),axis=1)
ax.add_collection(LineCollection(segments,colors='darkred',linewidths=1))
ax.scatter(p[:,0],p[:,1],s=35,c='orangered',label='DEM disks; centers shown')
a=np.linspace(0,2*np.pi,100);center=np.array(dem['tool_start_m'])+dem['time_s']*np.array(dem['tool_velocity_m_s'])
ax.plot(center[0]+.1*np.cos(a),center[1]+.1*np.sin(a),color='black',label='Prescribed circular tool')
ax.set_xlim(-.4,.4);ax.set_ylim(-.2,.35);ax.set_aspect('equal');ax.legend(fontsize=8)
ax.set_xlabel('Global x [m]');ax.set_ylabel('Global y [m]')
ax.set_title('Actual final fields at 0.15 s; slipping IBM, no free surface')
fig.savefig(out/'final-fields.png',dpi=180)
