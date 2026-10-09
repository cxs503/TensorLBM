"""Exact 2-D Cartesian cut-cell geometry for an interior translating rectangle.

SI lengths/time; areas represent volume per unit extrusion thickness. Wall
normals point out of liquid into solid. No fluid populations are advanced.
Space-time integration splits at every edge/grid crossing; midpoint is used
only to select topology, never to approximate an integral.
"""
from dataclasses import dataclass
import math
import numbers
import torch


def _real(value,name):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be a finite real')
    return float(value)


def _pair(value,name):
    if not isinstance(value,(tuple,list)) or len(value)!=2:
        raise ValueError(f'{name} must have two entries')
    return tuple(_real(v,name) for v in value)


@dataclass(frozen=True)
class SweptRectangleGeometry:
    nx: int = 12
    ny: int = 10
    dx_m: float = .1
    width_m: float = .31
    height_m: float = .27
    center_m: tuple = (.55,.48)
    velocity_m_s: tuple = (.08,.03)

    def __post_init__(self):
        if type(self.nx) is not int or type(self.ny) is not int or min(self.nx,self.ny)<2:
            raise ValueError('grid dimensions must be integer >= 2')
        for name in ('dx_m','width_m','height_m'):
            value=_real(getattr(self,name),name)
            if value<=0:raise ValueError(f'{name} must be positive')
            object.__setattr__(self,name,value)
        for name in ('center_m','velocity_m_s'):
            object.__setattr__(self,name,_pair(getattr(self,name),name))
        self.bounds(0.)

    def config(self):
        return dict(nx=self.nx,ny=self.ny,dx_m=self.dx_m,width_m=self.width_m,height_m=self.height_m,
                    center_m=list(self.center_m),velocity_m_s=list(self.velocity_m_s))

    def bounds(self,time_s):
        t=_real(time_s,'time_s')
        if t<0:raise ValueError('time_s must be nonnegative')
        cx=self.center_m[0]+self.velocity_m_s[0]*t
        cy=self.center_m[1]+self.velocity_m_s[1]*t
        a,b=cx-self.width_m/2,cx+self.width_m/2
        c,d=cy-self.height_m/2,cy+self.height_m/2
        if not (0<a<b<self.nx*self.dx_m and 0<c<d<self.ny*self.dx_m):
            raise ValueError('rectangle must remain strictly inside domain; periodic wrap unsupported')
        # Canonicalize only floating point representations of exact grid
        # coincidences (at most a few ulps), not geometric small-cell volume.
        edges=[]
        for value in (a,b,c,d):
            q=value/self.dx_m;nearest=round(q)
            if abs(q-nearest)<16*math.ulp(max(1.,abs(q))):value=nearest*self.dx_m
            edges.append(value)
        return tuple(edges)

    @staticmethod
    def _overlap(edges,lower,upper):
        return (torch.minimum(edges[1:],torch.tensor(upper,dtype=torch.float64))-
                torch.maximum(edges[:-1],torch.tensor(lower,dtype=torch.float64))).clamp_min(0)

    def _grids(self):
        return (torch.arange(self.nx+1,dtype=torch.float64)*self.dx_m,
                torch.arange(self.ny+1,dtype=torch.float64)*self.dx_m)

    def _edge_cell(self,coordinate,negative_side):
        q=coordinate/self.dx_m
        nearest=round(q)
        if abs(q-nearest)<16*math.ulp(max(1.,abs(q))):
            return nearest-1 if negative_side else nearest
        return math.floor(q)

    def at(self,time_s=0.):
        a,b,c,d=self.bounds(time_s);x,y=self._grids()
        ox=self._overlap(x,a,b);oy=self._overlap(y,c,d)
        wx=x[1:]-x[:-1];wy=y[1:]-y[:-1]
        volume=wy[:,None]*wx[None,:]-oy[:,None]*ox[None,:]
        vertical=wy[:,None]-oy[:,None]*((x>=a)&(x<=b))[None,:]
        horizontal=wx[None,:]-((y>=c)&(y<=d))[:,None]*ox[None,:]
        facets=[];wall=torch.zeros(self.ny,self.nx,2,dtype=torch.float64)
        for edge,normal,negative_side in ((a,(1.,0.),True),(b,(-1.,0.),False)):
            col=self._edge_cell(edge,negative_side)
            for row in range(self.ny):
                lo=max(float(y[row]),c);hi=min(float(y[row+1]),d)
                if hi>lo:
                    length=hi-lo;wall[row,col]+=torch.tensor(normal,dtype=torch.float64)*length
                    facets.append(dict(cell_yx=[row,col],normal_xy=list(normal),length_m=length,
                                       center_m=[edge,(lo+hi)/2]))
        for edge,normal,negative_side in ((c,(0.,1.),True),(d,(0.,-1.),False)):
            row=self._edge_cell(edge,negative_side)
            for col in range(self.nx):
                lo=max(float(x[col]),a);hi=min(float(x[col+1]),b)
                if hi>lo:
                    length=hi-lo;wall[row,col]+=torch.tensor(normal,dtype=torch.float64)*length
                    facets.append(dict(cell_yx=[row,col],normal_xy=list(normal),length_m=length,
                                       center_m=[(lo+hi)/2,edge]))
        cart=torch.stack((vertical[:,1:]-vertical[:,:-1],horizontal[1:]-horizontal[:-1]),dim=-1)
        return dict(time_s=float(time_s),fluid_volume_m2=volume,vertical_open_m=vertical,
                    horizontal_open_m=horizontal,wall_facets=facets,wall_normal_length_m=wall,
                    closure_residual_m=cart+wall)

    def events(self,time_start_s,dt_s):
        t0=_real(time_start_s,'time_start_s');dt=_real(dt_s,'dt_s')
        if dt<=0:raise ValueError('dt_s must be positive')
        self.bounds(t0);self.bounds(t0+dt)
        times=[t0,t0+dt]
        bounds0=self.bounds(0.)
        for origin,speed,count in ((bounds0[0],self.velocity_m_s[0],self.nx),
                                   (bounds0[1],self.velocity_m_s[0],self.nx),
                                   (bounds0[2],self.velocity_m_s[1],self.ny),
                                   (bounds0[3],self.velocity_m_s[1],self.ny)):
            if speed:
                for k in range(count+1):
                    t=(k*self.dx_m-origin)/speed
                    if t0<t<t0+dt:times.append(t)
        # Keep close but distinct events: suppressing them would lose area.
        return sorted(set(times))

    def interval(self,time_start_s=0.,dt_s=.1):
        events=self.events(time_start_s,dt_s);x,y=self._grids()
        iv=torch.zeros(self.ny,self.nx+1,dtype=torch.float64)
        ih=torch.zeros(self.ny+1,self.nx,dtype=torch.float64)
        wall=torch.zeros(self.ny,self.nx,2,dtype=torch.float64);facets=[]
        for ta,tb in zip(events[:-1],events[1:]):
            if tb<=ta:continue
            tm=(ta+tb)/2;h=tb-ta
            ba=self.bounds(ta);bb=self.bounds(tb);bm=self.bounds(tm)
            ox_a=self._overlap(x,ba[0],ba[1]);ox_b=self._overlap(x,bb[0],bb[1])
            oy_a=self._overlap(y,ba[2],ba[3]);oy_b=self._overlap(y,bb[2],bb[3])
            iv+=h*((y[1:]-y[:-1])[:,None]-.5*(oy_a+oy_b)[:,None]*((x>bm[0])&(x<bm[1]))[None,:])
            ih+=h*((x[1:]-x[:-1])[None,:]-((y>bm[2])&(y<bm[3]))[:,None]*.5*(ox_a+ox_b)[None,:])
            # Static faces coincident with rectangle edges are blocked;
            # moving coincidences are isolated zero-measure event endpoints.
            if self.velocity_m_s[0]==0:
                iv_delta=h*.5*(oy_a+oy_b)
                for edge in (bm[0],bm[1]):
                    matched=(x==edge)
                    iv[:,matched]-=iv_delta[:,None]
            if self.velocity_m_s[1]==0:
                ih_delta=h*.5*(ox_a+ox_b)
                for edge in (bm[2],bm[3]):
                    matched=(y==edge)
                    ih[matched,:]-=ih_delta[None,:]
            specifications=[(0,0,(1.,0.),True),(1,0,(-1.,0.),False),
                            (2,1,(0.,1.),True),(3,1,(0.,-1.),False)]
            for edge_index,axis,normal,negative_side in specifications:
                fixed=self._edge_cell(bm[edge_index],negative_side)
                grid=y if axis==0 else x
                lower,upper=(2,3) if axis==0 else (0,1)
                for k in range(len(grid)-1):
                    sample=[]
                    for bounds in (ba,bm,bb):
                        lo=max(float(grid[k]),bounds[lower]);hi=min(float(grid[k+1]),bounds[upper])
                        length=max(0.,hi-lo)
                        pos=[bounds[edge_index],(lo+hi)/2] if axis==0 else [(lo+hi)/2,bounds[edge_index]]
                        sample.append((length,pos))
                    integral=h*(sample[0][0]+sample[2][0])/2
                    if integral<=0:continue
                    row,col=(k,fixed) if axis==0 else (fixed,k)
                    wall[row,col]+=torch.tensor(normal,dtype=torch.float64)*integral
                    moment=[h*(sample[0][0]*sample[0][1][j]+4*sample[1][0]*sample[1][1][j]+sample[2][0]*sample[2][1][j])/6 for j in range(2)]
                    facets.append(dict(cell_yx=[row,col],normal_xy=list(normal),
                        integrated_length_m_s=integral,integrated_position_length_m2_s=moment,
                        wall_velocity_m_s=list(self.velocity_m_s),time_start_s=ta,time_end_s=tb))
        start=self.at(time_start_s);end=self.at(time_start_s+dt_s)
        change=end['fluid_volume_m2']-start['fluid_volume_m2']
        swept=torch.einsum('yxa,a->yx',wall,torch.tensor(self.velocity_m_s,dtype=torch.float64))
        cart=torch.stack((iv[:,1:]-iv[:,:-1],ih[1:]-ih[:-1]),dim=-1)
        return dict(schema='tensorlbm.swept-rectangle-geometry/1',config=self.config(),
            time_start_s=float(time_start_s),dt_s=float(dt_s),event_times_s=events,
            start=start,end=end,vertical_open_integral_m_s=iv,horizontal_open_integral_m_s=ih,
            wall_facets=facets,wall_normal_integral_m_s=wall,swept_volume_m2=swept,
            fluid_volume_change_m2=change,gcl_residual_m2=change-swept,
            closure_integral_residual_m_s=cart+wall)
