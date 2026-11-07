import re

p = 'scripts/run_ship_wind_campaign.py'
s = open(p).read()
s = s.replace('''    ap.add_argument("--cs", type=float, default=0.16)''',
              '''    ap.add_argument("--cs", type=float, default=0.16)
    ap.add_argument("--u-clip", type=float, default=0.25,
                    help="velocity clip used to keep the LBM stable")''')
s = s.replace('''        solver = WindLoadSolver(geom, units, cs_smag=args.cs,
                                wall_model=args.wall_model,
                                wall_feedback=args.wall_feedback,
                                normalise_outflow=args.normalise_outflow)''',
              '''        solver = WindLoadSolver(geom, units, cs_smag=args.cs,
                                wall_model=args.wall_model,
                                wall_feedback=args.wall_feedback,
                                normalise_outflow=args.normalise_outflow,
                                u_clip=args.u_clip)''')
s = s.replace('''        cs_smag=args.cs, wall_model=args.wall_model,
        wall_feedback=args.wall_feedback, normalise_outflow=args.normalise_outflow)''',
              '''        cs_smag=args.cs, wall_model=args.wall_model,
        wall_feedback=args.wall_feedback, normalise_outflow=args.normalise_outflow,
        u_clip=args.u_clip)''')
s = s.replace('''            "wall_model": args.wall_model, "wall_feedback": args.wall_feedback,''',
              '''            "wall_model": args.wall_model, "wall_feedback": args.wall_feedback,
            "u_clip": args.u_clip,''')
s = s.replace('''        s = solver
        k = s.shape[0] // 3''', '''        s = solver
        k = max(1, int(0.30 * s.shape[0]))''')
s = s.replace('''        record["fields"] = args.fields''',
              '''        record["fields"] = args.fields
    record["force_history"] = {
        "fx": [round(s.fx, 6) for s in solver.force_history],
        "fy": [round(s.fy, 6) for s in solver.force_history]}''')
open(p, 'w').write(s)

p = 'scripts/run_ship_wind_batch.py'
s = open(p).read()
s = s.replace('DX_POLAR = 3.0', 'DX_POLAR = 4.0')
s = s.replace('DX_COARSE = 4.5', 'DX_COARSE = 6.0')
s = s.replace('DX_FINE = 2.2', 'DX_FINE = 3.3')
s = s.replace('STEPS = {"3.0": (700, 350), "4.5": (600, 300), "2.2": (900, 450)}',
              'STEPS = {"4.0": (1200, 600), "6.0": (700, 350), "3.3": (1500, 800)}')
s = s.replace('''    ap.add_argument("--redo", action="store_true")''',
              '''    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--cs", type=float, default=0.20)
    ap.add_argument("--u-clip", type=float, default=0.40)
    ap.add_argument("--wall-model", action="store_true", default=False)''')
s = s.replace('''    return exe + ["--ship", t["ship"], "--angle", str(t["angle"]),
                  "--dx", str(t["dx"]), "--steps", str(steps),
                  "--avg-from", str(avg), "--threads", "1"] + margins(t["ship"], t["dx"]) + \\''',
              '''    extra = ["--cs", f"{CS:.2f}", "--u-clip", f"{U_CLIP:.2f}"]
    if not WALL_MODEL:
        extra += ["--no-wall-model", "--no-wall-feedback"]
    return exe + ["--ship", t["ship"], "--angle", str(t["angle"]),
                  "--dx", str(t["dx"]), "--steps", str(steps),
                  "--avg-from", str(avg), "--threads", "1"] + extra + \\
        margins(t["ship"], t["dx"]) + \\''')
s = s.replace('''def tasks(only: str) -> list[dict]:''',
              '''CS = 0.20
U_CLIP = 0.40
WALL_MODEL = False


def tasks(only: str) -> list[dict]:''')
s = s.replace('''    os.makedirs(RAW, exist_ok=True)''',
              '''    global CS, U_CLIP, WALL_MODEL
    CS, U_CLIP, WALL_MODEL = args.cs, args.u_clip, args.wall_model
    os.makedirs(RAW, exist_ok=True)''')
open(p, 'w').write(s)
print("patched both")