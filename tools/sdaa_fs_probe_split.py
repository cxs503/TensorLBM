#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SDAA（太初卡）free_surface_probe 包装器——**分相位**版（2026-11-21, K1-DIAG-B）。

与共享包装器 ``/root/tp-agent/cluster/sdaa_fs_probe.py`` 协议完全一致（同样的卡锁、
同样的容器镜像、同样的 driver、同样的产出路径），唯一差别是把「递交 → 轮询 → 回收」
拆成三个可独立调用的相位，因为共享包装器的单次调用会阻塞 >120 s（平台 shell 工具
的硬上限），在本机无法一次跑完。

用法::

    python3 tools/sdaa_fs_probe_split.py launch --arm NAME [--steps N] [--cells 8] \
        [--tree .] [--diag-cavity]
    python3 tools/sdaa_fs_probe_split.py poll   --arm NAME
    python3 tools/sdaa_fs_probe_split.py collect --arm NAME

产出：``<tree>/runs/<arm>/result.json`` + ``run.log``（含 wall/卡号/commit 溯源）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HOST = "test002@10.71.12.99"
CRED = "/root/.lws702_cred"
IMG = "lbm_test:snap_1114"


def ssh(args, stdin=None, timeout=120):
    cmd = ["timeout", str(timeout), "sshpass", "-f", CRED, "ssh",
           "-o", "StrictHostKeyChecking=no", HOST, args]
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout + 10)


DRIVER = r'''
import sys, os, time, json
from pathlib import Path
import torch
sys.path.insert(0, "/we"); sys.path.insert(0, "/we/src")
torch.set_num_threads(4)
import tensorlbm.d3q27
import torch_sdaa
torch.set_default_device("sdaa")
_on = torch.Tensor.numpy
def _shim(self, *ar, **kw):
    if self.device.type != "cpu": return _on(self.cpu(), *ar, **kw)
    return _on(self, *ar, **kw)
torch.Tensor.numpy = _shim
os.chdir("/we")
import importlib, inspect
fp = importlib.import_module("benchmarks.water_entry.single_drop.free_surface_probe")
t0 = time.time()
kw = dict(n=int(os.environ["WE_CELLS"]), speed=float(os.environ["WE_SPD"]), limit=int(os.environ["WE_STEPS"]),
    gap_fraction=float(os.environ["WE_GAP"]), capacity=True, nearest=False, pilot=False, span=4.0,
    pool_depth=float(os.environ["WE_PD"]), simple=True, release=float(os.environ["WE_RS"]),
    planes_path=None, silhouette_path=None, ledger=os.environ.get("WE_LEDGER", "0") == "1", diag_path=None,
    u_limit=float(os.environ["WE_UL"]))
_cav = float(os.environ.get("WE_CAV", "0") or 0)
if _cav: kw["cav_deficit"] = _cav
_cav_probe = os.environ.get("WE_CAV_PROBE", "0") == "1"
_cav_probe_out = os.environ.get("WE_CAV_PROBE_OUT", "cav_probe.json")
if _cav_probe:
    kw["cav_probe"] = True
    kw["cav_probe_out"] = Path(_cav_probe_out)
_curv2 = float(os.environ.get("WE_CURV2", "0") or 0)
if _curv2: kw["curv2_dose"] = _curv2
_stbf = os.environ.get("WE_STBF", "0") == "1"
if os.environ.get("WE_DIAG_CAVITY", "0") == "1": kw["diag_cavity"] = True
if os.environ.get("WE_DIAG_BUDGET", "0") == "1": kw["diag_budget"] = True
_momconv = os.environ.get("WE_MOMCONV", "") or None
if _momconv: kw["momconv"] = _momconv
if _stbf: kw["st_bodyforce"] = True
_film = os.environ.get("WE_FILM", "0") == "1"
if _film: kw["cavity_wall_film"] = True
if os.environ.get("WE_PLANES", "0") == "1": kw["planes_path"] = Path("/out/planes.npz")
_sig = inspect.signature(fp.probe)
result = fp.probe(**{k: v for k, v in kw.items() if k in _sig.parameters})
result["_wall_s"] = time.time() - t0
json.dump(result, open("/out/result.json", "w"))
h = result.get("history") or []
pk = max(h, key=lambda r_: r_.get("depth_over_D", -1)) if h else {}
print("SDAA_DONE stop=%s peak=%.4fD@%sms wall=%.0fs" % (result.get("stop_reason"), pk.get("depth_over_D", -1), pk.get("time_ms", -1), result["_wall_s"]), flush=True)
'''


def state_path(tree, arm):
    return os.path.join(tree, "runs", arm, "_sdaa_state.json")


def load_state(tree, arm):
    with open(state_path(tree, arm)) as handle:
        return json.load(handle)


def save_state(tree, arm, state):
    path = state_path(tree, arm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(state, handle, indent=1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["launch", "poll", "collect"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tree", default=".")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--card", type=int, default=-1)
    ap.add_argument("--gap", type=float, default=0.18)
    ap.add_argument("--speed", type=float, default=0.06)
    ap.add_argument("--rs", type=float, default=1.53)
    ap.add_argument("--pd", type=float, default=3.0)
    ap.add_argument("--ul", type=float, default=0.15)
    ap.add_argument("--diag-cavity", action="store_true",
                    help="enable the read-only cavity/drop diagnosis inside the probe")
    ap.add_argument("--planes", action="store_true",
                    help="dump the mid-plane fill/velocity at the registered post-contact times")
    ap.add_argument("--diag-budget", action="store_true",
                    help="enable the read-only momentum/energy budget ledger inside the probe")
    ap.add_argument("--momconv", default="",
                    choices=["", "inherit", "redistribute", "redistribute_liq"],
                    help="momentum-carrying I->G conversion (DE-X-K3-MOMCONV); empty=legacy")
    return ap.parse_args()


def launch(a):
    tree = os.path.abspath(a.tree)
    assert os.path.isdir(f"{tree}/src/tensorlbm") and os.path.isdir(f"{tree}/benchmarks/water_entry"), \
        "need src/tensorlbm + benchmarks in tree"
    head = subprocess.run(["git", "-C", tree, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "dirty"
    tag = f"{a.arm}_{int(time.time())}"
    wt = f"/tmp/we_{tag}"
    out = f"/tmp/we_out_{tag}"

    tar = subprocess.run(["tar", "czf", "-", "-C", tree, "src", "benchmarks"], capture_output=True)
    assert tar.returncode == 0, tar.stderr.decode()[:200]
    r = ssh(f"mkdir -p {wt} {out} && tar xzf - -C {wt} && mkdir -p /tmp/sdaa_locks",
            stdin=tar.stdout, timeout=180)
    assert r.returncode == 0, r.stderr.decode()[-300:]

    card = a.card
    if card < 0:
        for c in range(32):
            r = ssh("mkdir /tmp/sdaa_locks/card%d 2>/dev/null && echo OK" % c)
            if r.returncode == 0 and b"OK" in r.stdout:
                card = c
                break
        assert card >= 0, "32 cards busy"
    else:
        r = ssh("mkdir /tmp/sdaa_locks/card%d 2>/dev/null && echo OK" % card)
        assert r.returncode == 0 and b"OK" in r.stdout, "card %d busy" % card

    r = ssh(f"cat > {wt}/probe_wrap.py", stdin=DRIVER.encode())
    assert r.returncode == 0, r.stderr.decode()[-200:]

    env = (f"-e WE_DEV=sdaa -e WE_CELLS={a.cells} -e WE_STEPS={a.steps} -e WE_SPD={a.speed} "
           f"-e WE_GAP={a.gap} -e WE_PD={a.pd} -e WE_RS={a.rs} -e WE_UL={a.ul} "
           f"-e WE_DIAG_CAVITY={1 if a.diag_cavity else 0} -e WE_PLANES={1 if a.planes else 0} "
           f"-e WE_DIAG_BUDGET={1 if a.diag_budget else 0} -e WE_MOMCONV={a.momconv}")
    cname = f"sdaa_{tag}"
    r = ssh(f"docker run -d --name {cname} --device /dev/tcaicard{card} "
            f"-v {wt}:/we -v {out}:/out {env} {IMG} bash -c "
            f"'source /opt/tecoai/setvars.sh 2>/dev/null; python3 /we/probe_wrap.py'")
    assert r.returncode == 0, r.stderr.decode()[-300:]

    save_state(tree, a.arm, {"arm": a.arm, "card": card, "container": cname, "wt": wt,
                             "out": out, "commit": head, "steps": a.steps, "cells": a.cells,
                             "diag_cavity": bool(a.diag_cavity), "planes": bool(a.planes),
                             "diag_budget": bool(a.diag_budget),
                             "momconv": a.momconv,
                             "launched_at": time.time()})
    print(f"[sdaa] launched arm={a.arm} card={card} commit={head} steps={a.steps} "
          f"diag_cavity={bool(a.diag_cavity)} diag_budget={bool(a.diag_budget)} "
          f"momconv={a.momconv!r} container={cname}", flush=True)
    return 0


def poll(a):
    tree = os.path.abspath(a.tree)
    state = load_state(tree, a.arm)
    cname = state["container"]
    r = ssh(f"docker ps -a --filter name={cname} --format '{{{{.Status}}}}'; "
            f"docker logs --tail 3 {cname} 2>&1 | tail -3")
    print(r.stdout.decode(errors="ignore"), flush=True)
    return 0


def collect(a):
    tree = os.path.abspath(a.tree)
    state = load_state(tree, a.arm)
    cname, wt, out, card = state["container"], state["wt"], state["out"], state["card"]
    os.makedirs(f"{tree}/runs/{a.arm}", exist_ok=True)
    rlog = ssh(f"docker logs {cname} 2>&1 | tail -120")
    rj = ssh(f"cat {out}/result.json 2>/dev/null")
    rp = ssh(f"cat {out}/planes.npz 2>/dev/null") if state.get("planes") else None
    header = (f"arm={a.arm} card={card} commit={state['commit']} steps={state['steps']} "
              f"diag_cavity={state['diag_cavity']} diag_budget={state.get('diag_budget', False)} "
              f"momconv={state.get('momconv', '')}\n")
    with open(f"{tree}/runs/{a.arm}/run.log", "w") as handle:
        handle.write(header + rlog.stdout.decode(errors="ignore"))
    if rp is not None and rp.stdout:
        with open(f"{tree}/runs/{a.arm}/planes.npz", "wb") as handle:
            handle.write(rp.stdout)
    ok = False
    try:
        j = json.loads(rj.stdout.decode(errors="ignore").strip())
        with open(f"{tree}/runs/{a.arm}/result.json", "w") as handle:
            json.dump(j, handle)
        ok = True
    except Exception as error:  # noqa: BLE001
        print(f"[sdaa] result.json parse failed: {error}")
    ssh(f"docker rm -f {cname} >/dev/null 2>&1; rm -rf {wt} {out}; "
        f"rmdir /tmp/sdaa_locks/card{card} 2>/dev/null; true")
    print(f"[sdaa] collected arm={a.arm} result_json={'OK' if ok else 'MISSING'} "
          f"→ runs/{a.arm}/", flush=True)
    return 0 if ok else 1


def main():
    a = parse_args()
    return {"launch": launch, "poll": poll, "collect": collect}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
