import pathlib
p = pathlib.Path("benchmarks/p40w4_n04_nonnewtonian_rheology/run.py")
s = p.read_text()
subs = [
('''    out, ref = run_case(h, closure, u_max_target, accel=accel)
    met = profile_metrics(out, ref, y=None)
    plug = plug_report(''',
'''    out, ref = run_case(h, closure, u_max_target, accel=accel, n_steps=cfg["n_steps"],
                        tol=cfg["tol"], check_every=cfg["check_every"])
    met = profile_metrics(out, ref, y=None)
    plug = plug_report('''),
('''def run_case(h: int, closure, u_max_target: float = 0.02, accel: float | None = None,
             nx: int = 4, n_steps: int = 200_000, tol: float = 1e-7,
             check_every: int = 500) -> tuple[dict, dict]:''',
'''def run_case(h: int, closure, u_max_target: float = 0.02, accel: float | None = None,
             nx: int = 4, n_steps: int = 120_000, tol: float = 1e-5,
             check_every: int = 500) -> tuple[dict, dict]:'''),
('''    closure = newtonian(0.1)
    out, ref = run_case(h, closure, 0.02)''',
'''    closure = newtonian(0.1)
    out, ref = run_case(h, closure, 0.02, n_steps=cfg["n_steps"], tol=cfg["tol"],
                        check_every=cfg["check_every"])'''),
('''def power_law_case(h: int, n: float, u_max_target: float = 0.02, nu_wall: float = 0.1) -> dict:''',
'''def power_law_case(h: int, n: float, u_max_target: float = 0.02, nu_wall: float = 0.1,
                   n_steps: int = 120_000, tol: float = 1e-5, check_every: int = 500) -> dict:'''),
('''    out, ref = run_case(h, closure, u_max_target, accel=accel)
    rows = out["fluid_rows"]''',
'''    out, ref = run_case(h, closure, u_max_target, accel=accel, n_steps=n_steps,
                        tol=tol, check_every=check_every)
    rows = out["fluid_rows"]'''),
('''        res = power_law_case(cfg["h_powerlaw"], n)''',
'''        res = power_law_case(cfg["h_powerlaw"], n, n_steps=cfg["n_steps"], tol=cfg["tol"],
                             check_every=cfg["check_every"])'''),
('''        res = power_law_case(h, n)''',
'''        res = power_law_case(h, n, n_steps=cfg["n_steps"], tol=cfg["tol"],
                             check_every=cfg["check_every"])'''),
('''    closure = carreau_yasuda(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0,
                                 flow_index=0.4, yasuda_exponent=2.0)
    else:
        closure = cross(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0, exponent=1.2)
    out, ref = run_case(h, closure, 0.02)''',
'''    closure = carreau_yasuda(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0,
                                 flow_index=0.4, yasuda_exponent=2.0)
    else:
        closure = cross(nu_0=0.30, nu_inf=0.02, relaxation_time=500.0, exponent=1.2)
    out, ref = run_case(h, closure, 0.02, n_steps=cfg["n_steps"], tol=cfg["tol"],
                        check_every=cfg["check_every"])'''),
('''    "n_steps": 200_000,
    "tol": 1e-7,
    "check_every": 500,''', '''    "n_steps": 120_000,
    "tol": 1e-5,
    "check_every": 500,'''),
('''    "n_steps": 40_000,
    "tol": 1e-6,
    "check_every": 200,''', '''    "n_steps": 15_000,
    "tol": 1e-4,
    "check_every": 250,'''),
]
for old, new in subs:
    assert old in s, old[:70]
    s = s.replace(old, new, 1)
p.write_text(s)
print("patched")
