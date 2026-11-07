#!/usr/bin/env python3
"""Batch launcher for the IN38-N04 LBM campaign (bounded concurrency).

Builds the task list (ship x angle x grid), runs each case as an isolated
one-thread process with ``nice``, and keeps at most ``--workers`` in flight.
Each case writes ``runs/in38_n04/raw/<tag>.json``; failures are recorded in
``runs/in38_n04/raw/_log.txt`` so a re-run only repeats missing cases.

    python3 scripts/run_ship_wind_batch.py --workers 7            # everything
    python3 scripts/run_ship_wind_batch.py --only polar            # main polar
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "runs", "in38_n04", "raw")

ANGLES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
SHIPS = ("container", "tanker", "bulker")
#: superstructure height per catalogue entry, used to scale the margins
HEIGHT = {"container": 33.0, "tanker": 33.0, "bulker": 31.0}
DX_POLAR = 4.0
DX_COARSE = 6.0
DX_FINE = 3.3
STEPS = {"4.0": (1200, 600), "6.0": (700, 350), "3.3": (1500, 800)}


def margins(ship: str, dx: float) -> list[str]:
    h = HEIGHT[ship]
    up, down, side, top = 2.0 * h, 4.0 * h, 2.5 * h, 2.5 * h
    return ["--margin-upstream", f"{up:.1f}", "--margin-downstream", f"{down:.1f}",
            "--margin-side", f"{side:.1f}", "--margin-top", f"{top:.1f}"]


def tag(ship: str, angle: float, dx: float) -> str:
    return f"{ship}_a{int(angle):03d}_dx{dx:.1f}"


CS = 0.20
U_CLIP = 0.40
WALL_MODEL = False


def tasks(only: str) -> list[dict]:
    out: list[dict] = []
    if only in ("all", "polar"):
        for ship in SHIPS:
            for a in ANGLES:
                out.append({"tag": tag(ship, a, DX_POLAR), "ship": ship,
                            "angle": a, "dx": DX_POLAR, "kind": "polar"})
    if only in ("all", "gci"):
        for ship in SHIPS:
            for a in (0.0, 90.0):
                out.append({"tag": tag(ship, a, DX_COARSE), "ship": ship,
                            "angle": a, "dx": DX_COARSE, "kind": "gci"})
        for a in (0.0, 90.0):
            out.append({"tag": tag("container", a, DX_FINE), "ship": "container",
                        "angle": a, "dx": DX_FINE, "kind": "gci"})
    if only in ("all", "cube"):
        for cells, re in ((8, 3000.0), (8, 20000.0), (12, 20000.0)):
            out.append({"tag": f"cube_re{int(re)}_n{cells}", "cube": True,
                        "cells": cells, "re": re, "kind": "cube"})
    return out


def cmd_for(t: dict) -> list[str]:
    exe = [sys.executable, os.path.join(HERE, "scripts",
                                        "run_ship_wind_campaign.py")]
    if t.get("cube"):
        return exe + ["--case", "cube", "--cells", str(t["cells"]),
                      "--re", str(t["re"]), "--steps", "900", "--avg-from", "450",
                      "--domain", "10.0,4.0,4.0", "--threads", "1", "--out",
                      os.path.join(RAW, t["tag"] + ".json")]
    steps, avg = STEPS[f"{t['dx']:.1f}"]
    extra = ["--cs", f"{CS:.2f}", "--u-clip", f"{U_CLIP:.2f}"]
    if not WALL_MODEL:
        extra += ["--no-wall-model", "--no-wall-feedback"]
    return exe + ["--ship", t["ship"], "--angle", str(t["angle"]),
                  "--dx", str(t["dx"]), "--steps", str(steps),
                  "--avg-from", str(avg), "--threads", "1"] + extra + \
        margins(t["ship"], t["dx"]) + \
        ["--out", os.path.join(RAW, t["tag"] + ".json"),
         "--fields", os.path.join(RAW, t["tag"] + "_fields.npz")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--only", default="all",
                    choices=["all", "polar", "gci", "cube"])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--cs", type=float, default=0.20)
    ap.add_argument("--u-clip", type=float, default=0.40)
    ap.add_argument("--wall-model", action="store_true", default=False)
    args = ap.parse_args()

    global CS, U_CLIP, WALL_MODEL
    CS, U_CLIP, WALL_MODEL = args.cs, args.u_clip, args.wall_model
    os.makedirs(RAW, exist_ok=True)
    log = os.path.join(RAW, "_log.txt")
    todo = [t for t in tasks(args.only)
            if args.redo or not os.path.exists(os.path.join(RAW, t["tag"] + ".json"))]
    print(f"{len(todo)} cases to run, {args.workers} workers", flush=True)

    running: list[tuple[subprocess.Popen, dict, float]] = []
    done = failed = 0
    queue = list(todo)
    t_start = time.time()
    while queue or running:
        while queue and len(running) < args.workers:
            t = queue.pop(0)
            fh = open(os.path.join(RAW, t["tag"] + ".log"), "w")
            p = subprocess.Popen(["nice", "-n", "10"] + cmd_for(t),
                                 stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, t, time.time()))
            print(f"[start] {t['tag']}", flush=True)
        time.sleep(3.0)
        for item in list(running):
            p, t, t0 = item
            if p.poll() is not None:
                running.remove(item)
                ok = p.returncode == 0 and os.path.exists(
                    os.path.join(RAW, t["tag"] + ".json"))
                if ok:
                    done += 1
                else:
                    failed += 1
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"{t['tag']} rc={p.returncode} "
                             f"{'ok' if ok else 'FAIL'} "
                             f"{time.time() - t0:.0f}s\n")
                print(f"[{'done' if ok else 'FAIL'}] {t['tag']} "
                      f"{time.time() - t0:.0f}s ({done} ok, {failed} fail, "
                      f"{time.time() - t_start:.0f}s elapsed)", flush=True)
    print(f"finished: {done} ok, {failed} failed, "
          f"{time.time() - t_start:.0f}s", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())