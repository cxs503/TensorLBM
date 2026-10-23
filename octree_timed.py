"""用计时钩子定位 octree 极慢阶段 (SDAA)。"""
import sys, time
sys.path.insert(0, 'src')
import torch
import tensorlbm.static_block_amr as sba

# 保存原始方法
_orig_step = sba.NestedStaticBlockAMR3D.step
_orig_fill = sba.StaticBlockAMR3D._fill_ghost
_orig_restrict = sba.StaticBlockAMR3D._restrict_physical
_orig_advance_iface = sba.NestedStaticBlockAMR3D._advance_interface

# 计时包装
def timed_step(self, *args, **kw):
    t0 = time.time()
    r = _orig_step(self, *args, **kw)
    dt = time.time() - t0
    if not hasattr(self, '_step_times'):
        self._step_times = []
    self._step_times.append(dt)
    if len(self._step_times) <= 3 or len(self._step_times) % 5 == 0:
        print(f'[timed] step #{len(self._step_times)} total={dt:.3f}s', flush=True)
    return r

def timed_fill(self, *args, **kw):
    t0 = time.time()
    r = _orig_fill(self, *args, **kw)
    dt = time.time() - t0
    if not hasattr(self, '_fill_times'):
        self._fill_times = []
    self._fill_times.append(dt)
    if dt > 0.5:
        print(f'[timed]   _fill_ghost={dt:.3f}s (SLOW)', flush=True)
    return r

def timed_restrict(self, *args, **kw):
    t0 = time.time()
    r = _orig_restrict(self, *args, **kw)
    dt = time.time() - t0
    if not hasattr(self, '_restrict_times'):
        self._restrict_times = []
    self._restrict_times.append(dt)
    if dt > 0.5:
        print(f'[timed]   _restrict={dt:.3f}s (SLOW)', flush=True)
    return r

def timed_advance_iface(self, *args, **kw):
    t0 = time.time()
    r = _orig_advance_iface(self, *args, **kw)
    dt = time.time() - t0
    if not hasattr(self, '_iface_times'):
        self._iface_times = []
    self._iface_times.append(dt)
    if dt > 0.5:
        print(f'[timed]   _advance_interface={dt:.3f}s (SLOW)', flush=True)
    return r

sba.NestedStaticBlockAMR3D.step = timed_step
sba.StaticBlockAMR3D._fill_ghost = timed_fill
sba.StaticBlockAMR3D._restrict_physical = timed_restrict
sba.NestedStaticBlockAMR3D._advance_interface = timed_advance_iface

# 计时 octree 壳层: fill_ghost / stream_gather / bfl / collide
import tensorlbm.octree_boundary.stepping as ost
_orig_fill_ghost_oct = ost.fill_ghost
_orig_stream_gather = ost.stream_gather

def timed_oct_fill(*args, **kw):
    t0 = time.time()
    r = _orig_fill_ghost_oct(*args, **kw)
    dt = time.time() - t0
    if dt > 0.3:
        print(f'[timed]   oct fill_ghost={dt:.3f}s (SLOW)', flush=True)
    return r

def timed_stream_gather(*args, **kw):
    t0 = time.time()
    r = _orig_stream_gather(*args, **kw)
    dt = time.time() - t0
    if dt > 0.3:
        print(f'[timed]   oct stream_gather={dt:.3f}s (SLOW)', flush=True)
    return r

ost.fill_ghost = timed_oct_fill
ost.stream_gather = timed_stream_gather

# 现在跑 octree 脚本
import runpy
sys.argv = ['octree_sphere_validate.py', '--device', 'sdaa:0', '--nx', '96', '--ny', '64', '--nz', '64',
            '--radius', '6', '--reynolds', '100', '--steps', '10', '--warmup-steps', '2', '--ramp-steps', '2',
            '--report-interval', '5', '--output', '/tmp/octree_timed.json']
runpy.run_path('examples/octree_sphere_validate.py', run_name='__main__')
