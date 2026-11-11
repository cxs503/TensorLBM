import importlib, os, sys, traceback, json, pkgutil
sys.path.insert(0, '/root/tp-agent/work/m14806/src')
import tensorlbm
print('pkg at', tensorlbm.__file__)
names = sorted(m.name for m in pkgutil.iter_modules(tensorlbm.__path__))
fails = []
ok = 0
for n in names:
    full = 'tensorlbm.' + n
    try:
        importlib.import_module(full)
        ok += 1
    except Exception as e:
        fails.append((n, type(e).__name__, str(e).splitlines()[0][:160]))
print('modules', len(names), 'import ok', ok, 'failed', len(fails))
for f in fails:
    print('|'.join(f))
json.dump({'n': len(names), 'ok': ok, 'fails': fails}, open('/root/tp-agent/work/m14806/artifacts/_import_sweep.json', 'w'), indent=1)