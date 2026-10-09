"""Independent geometry raw-field audit: no solver/geometry module imported."""
import gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/assets/swept-rectangle-geometry'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(lo,hi,a,b):return max(0.,min(hi,b)-max(lo,a))
def bounds(cfg,t):
    cx=cfg['center_m'][0]+cfg['velocity_m_s'][0]*t;cy=cfg['center_m'][1]+cfg['velocity_m_s'][1]*t
    return cx-cfg['width_m']/2,cx+cfg['width_m']/2,cy-cfg['height_m']/2,cy+cfg['height_m']/2

def inside(value,lower,upper):
    eps=16*math.ulp(max(1.,abs(value),abs(lower),abs(upper)))
    return lower-eps<=value<=upper+eps

def reference_faces(cfg,t):
    nx,ny,dx=cfg['nx'],cfg['ny'],cfg['dx_m'];a,b,c,d=bounds(cfg,t)
    vertical=[[(j+1)*dx-j*dx-overlap(j*dx,(j+1)*dx,c,d)*inside(k*dx,a,b) for k in range(nx+1)] for j in range(ny)]
    horizontal=[[(k+1)*dx-k*dx-overlap(k*dx,(k+1)*dx,a,b)*inside(j*dx,c,d) for k in range(nx)] for j in range(ny+1)]
    return vertical,horizontal

def main():
    study=json.loads((OUT/'study.json').read_text())
    for item in study['cases']:
        path=OUT/item['artifact'];assert sha(path)==item['sha256'];raw=json.loads(gzip.decompress(path.read_bytes()));cfg=raw['config']
        assert raw['source_sha256']==sha(ROOT/'src/tensorlbm/swept_rectangle_geometry_2d.py')
        assert raw['runner_sha256']==sha(ROOT/'examples/swept_rectangle_geometry/run.py')
        nx,ny,dx=cfg['nx'],cfg['ny'],cfg['dx_m'];vx,vy=cfg['velocity_m_s'];area=nx*ny*dx*dx-cfg['width_m']*cfg['height_m']
        births=deaths=0;maxgcl=maxclosure=maxarea=0.
        for h in raw['history']:
            t0=h['time_start_s'];dt=h['dt_s'];assert h['event_times_s'][0]==t0 and h['event_times_s'][-1]==t0+dt
            for snap in (h['start'],h['end']):
                a,b,c,d=bounds(cfg,snap['time_s']);total=0.
                for j in range(ny):
                    for k in range(nx):
                        reference=((k+1)*dx-k*dx)*((j+1)*dx-j*dx)-overlap(k*dx,(k+1)*dx,a,b)*overlap(j*dx,(j+1)*dx,c,d)
                        actual=snap['fluid_volume_m2'][j][k]
                        assert actual>=0 and abs(actual-reference)<1e-12*dx*dx
                        total+=actual
                        normals=snap['wall_normal_length_m'][j][k]
                        closure=[snap['vertical_open_m'][j][k+1]-snap['vertical_open_m'][j][k]+normals[0],snap['horizontal_open_m'][j+1][k]-snap['horizontal_open_m'][j][k]+normals[1]]
                        assert max(map(abs,closure))<1e-12*dx
                maxarea=max(maxarea,abs(total-area))
            # Two-point Gauss is independently exact for each linear aperture
            # after edge/corner event splitting; it does not use stored integrals.
            iv=[[0.]*(nx+1) for _ in range(ny)];ih=[[0.]*nx for _ in range(ny+1)]
            for ta,tb in zip(h['event_times_s'][:-1],h['event_times_s'][1:]):
                half=(tb-ta)/2;mid=(ta+tb)/2
                for t in [mid-half/math.sqrt(3),mid+half/math.sqrt(3)]:
                    rv,rh=reference_faces(cfg,t)
                    for j in range(ny):
                        for k in range(nx+1):iv[j][k]+=half*rv[j][k]
                    for j in range(ny+1):
                        for k in range(nx):ih[j][k]+=half*rh[j][k]
            fw=[[[0.,0.] for _ in range(nx)] for _ in range(ny)];perimeter=moment=0.
            for f in h['wall_facets']:
                j,k=f['cell_yx'];n=f['normal_xy'];length=f['integrated_length_m_s']
                assert length>0 and tuple(n) in {(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)}
                perimeter+=length
                for a in range(2):
                    fw[j][k][a]+=n[a]*length
                    moment+=n[a]*f['integrated_position_length_m2_s'][a]
            assert abs(perimeter-2*(cfg['width_m']+cfg['height_m'])*dt)<1e-12*dx*dt
            assert abs(moment+2*cfg['width_m']*cfg['height_m']*dt)<1e-12*dx*dx*dt
            for j in range(ny):
                for k in range(nx):
                    for a in range(2):assert abs(fw[j][k][a]-h['wall_normal_integral_m_s'][j][k][a])<1e-12*dx*dt
                    change=h['end']['fluid_volume_m2'][j][k]-h['start']['fluid_volume_m2'][j][k]
                    swept=vx*fw[j][k][0]+vy*fw[j][k][1]
                    maxgcl=max(maxgcl,abs(change-swept))
                    assert abs(change-h['fluid_volume_change_m2'][j][k])<1e-14*dx*dx
                    assert abs(swept-h['swept_volume_m2'][j][k])<1e-14*dx*dx
                    closure=[iv[j][k+1]-iv[j][k]+fw[j][k][0],ih[j+1][k]-ih[j][k]+fw[j][k][1]]
                    maxclosure=max(maxclosure,*map(abs,closure))
                    va=h['start']['fluid_volume_m2'][j][k];vb=h['end']['fluid_volume_m2'][j][k]
                    births+=int(va==0 and vb>0);deaths+=int(va>0 and vb==0)
                for k in range(nx+1):assert abs(iv[j][k]-h['vertical_open_integral_m_s'][j][k])<1e-12*dx*dt
            for j in range(ny+1):
                for k in range(nx):assert abs(ih[j][k]-h['horizontal_open_integral_m_s'][j][k])<1e-12*dx*dt
        assert maxgcl<item['gcl_threshold_m2'] and maxclosure<item['closure_threshold_m_s'] and maxarea<item['area_threshold_m2']
        assert births==item['birth_events'] and deaths==item['death_events'] and item['passed']
        print(item['case'],'independent area/face quadrature/closure/GCL PASS','birth/death',births,deaths)
    assert study['fluid_transport_implemented'] is False and study['physical_accuracy_qualified'] is False
if __name__=='__main__':main()
