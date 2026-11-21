"""Phase-split copy of /root/tp-agent/cluster/sdaa_fs_probe.py (RELX1-PHASETAU lane).

Why a local copy: the shared wrapper blocks for the whole run (poll loop),
which exceeds the agent tool's 120 s shell cap; backgrounding (nohup/&) is
hard-blocked by the local compute gate.  MMC5/MMC6 set the precedent of a
worktree wrapper copy (see X/MMC5-CURVMOM-B.md, X/MMC6-DYNANGLE-B.md).
Protocol is unchanged: same card lock (/tmp/sdaa_locks/card<N>, 32 cards),
same container driver, same env, same runs/<arm>/{result.json,run.log}
layout; run.log first line still records arm/card/commit/steps.

Usage:
  python3 tools/sdaa_fs_probe_pt.py launch --arm NAME [--steps N] [--cells 8] [--tree .]
  python3 tools/sdaa_fs_probe_pt.py poll   --arm NAME
  python3 tools/sdaa_fs_probe_pt.py collect --arm NAME
"""
import argparse, json, os, subprocess, sys, time

HOST = "test002@10.71.12.99"
CRED = "/root/.lws702_cred"
IMG = "lbm_test:snap_1114"


def ssh(args, stdin=None, timeout=120):
    cmd = ["timeout", str(timeout), "sshpass", "-f", CRED, "ssh",
           "-o", "StrictHostKeyChecking=no", HOST, args]
    return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout + 10)


def state_path(tree, arm):
    d = os.path.join(os.path.abspath(tree), "tmp")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"sdaa_pt_state_{arm}.json")


def load_state(tree, arm):
    p = state_path(tree, arm)
    if not os.path.exists(p):
        sys.exit(f"no state for arm {arm}; run launch first")
    return json.loads(open(p).read())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    for name in ("launch", "poll", "collect"):
        sp = sub.add_parser(name)
        sp.add_argument("--arm", required=True)
        sp.add_argument("--tree", default=".")
        sp.add_argument("--steps", type=int, default=10000)
        sp.add_argument("--cells", type=int, default=8)
        sp.add_argument("--card", type=int, default=-1)
        sp.add_argument("--gap", type=float, default=0.18)
        sp.add_argument("--speed", type=float, default=0.06)
        sp.add_argument("--rs", type=float, default=1.53)
        sp.add_argument("--pd", type=float, default=3.0)
        sp.add_argument("--ul", type=float, default=0.15)
    a = ap.parse_args()

    tree = os.path.abspath(a.tree)
    assert os.path.isdir(f"{tree}/src/tensorlbm") and os.path.isdir(f"{tree}/benchmarks/water_entry"), \
        "need src/tensorlbm + benchmarks in tree"
    head = subprocess.run(["git", "-C", tree, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "dirty"
    tag = f"{a.arm}_{int(time.time())}"
    wt = f"/tmp/we_{tag}"
    out = f"/tmp/we_out_{tag}"

    if a.phase == "launch":
        tar = subprocess.run(["tar", "czf", "-", "-C", tree, "src", "benchmarks"],
                            capture_output=True)
        assert tar.returncode == 0, tar.stderr.decode()[:200]
        r = ssh(f"mkdir -p {wt} {out} && tar xzf - -C {wt} && mkdir -p /tmp/sdaa_locks",
                  stdin=tar.stdout, timeout=180)
        assert r.returncode == 0, r.stderr.decode()[-300:]

        card = a.card
        if card < 0:
            for c in range(32):
                r = ssh(f"mkdir /tmp/sdaa_locks/card{c} 2>/dev/null && echo OK")
                if r.returncode == 0 and b"OK" in r.stdout:
                    card = c
                    break
            assert card >= 0, "32 cards all busy"
        else:
            r = ssh(f"mkdir /tmp/sdaa_locks/card{card} 2>/dev/null && echo OK")
            assert b"OK" in r.stdout, f"card {card} busy"

        driver = '''
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
_stbf = os.environ.get("WE_STBF", "0") == "1"
if _stbf: kw["st_bodyforce"] = True
_sig = inspect.signature(fp.probe)
result = fp.probe(**{k: v for k, v in kw.items() if k in _sig.parameters})
result["_wall_s"] = time.time() - t0
json.dump(result, open("/out/result.json", "w"))
h = result.get("history") or []
pk = max(h, key=lambda r_: r_.get("depth_over_D", -1)) if h else {}
print("SDAA_DONE stop=%s peak=%.4fD@%sms wall=%.0fs" % (result.get("stop_reason"), pk.get("depth_over_D", -1), pk.get("time_ms", -1), result["_wall_s"]), flush=True)
'''
        r = ssh(f"cat > {wt}/probe_wrap.py", stdin=driver.encode())
        assert r.returncode == 0, r.stderr.decode()[-300:]

        env = (f"-e WE_DEV=sdaa -e WE_CELLS={a.cells} -e WE_STEPS={a.steps} -e WE_SPD={a.speed} "
               f"-e WE_GAP={a.gap} -e WE_PD={a.pd} -e WE_RS={a.rs} -e WE_UL={a.ul} "
               f"-e WE_CAV=0 -e WE_STBF=0 -e WE_LEDGER=0")
        cname = f"sdaa_{tag}"
        r = ssh(f"docker run -d --name {cname} --device /dev/tcaicard{card} "
                f"-v {wt}:/we -v {out}:/out {env} {IMG} bash -c 'source /opt/tecoai/setvars.sh 2>/dev/null; python3 /we/probe_wrap.py'")
        assert r.returncode == 0, r.stderr.decode()[-300:]
        json.dump(dict(arm=a.arm, card=card, container=cname, wt=wt, out=out,
                       commit=head, steps=a.steps, tag=tag),
                  open(state_path(tree, a.arm), "w"))
        print(json.dumps(dict(arm=a.arm, card=card, commit=head, steps=a.steps,
                              container=cname)))
        return 0

    st = load_state(tree, a.arm)
    cname, wt, out = st["container"], st["wt"], st["out"]

    if a.phase == "poll":
        r = ssh(f"docker ps -a --filter name={cname} --format '{{{{.Status}}}}' && docker logs --tail 3 {cname} 2>&1 | tail -3")
        log_tail = r.stdout.decode(errors="ignore")
        done = "Exited" in log_tail
        print(json.dumps(dict(done=done, tail=log_tail.strip().splitlines()[-1][:160] if log_tail.strip() else "")))
        return 0

    # collect
    os.makedirs(f"{tree}/runs/{a.arm}", exist_ok=True)
    rlog = ssh(f"docker logs {cname} 2>&1 | tail -80")
    rj = ssh(f"cat {out}/result.json 2>/dev/null")
    open(f"{tree}/runs/{a.arm}/run.log", "w").write(
        f"arm={a.arm} card={st['card']} commit={st['commit']} steps={st['steps']}\n"
        + rlog.stdout.decode(errors="ignore"))
    ok = False
    try:
        j = json.loads(rj.stdout.decode(errors="ignore").strip())
        json.dump(j, open(f"{tree}/runs/{a.arm}/result.json", "w"))
        ok = True
    except Exception:
        pass
    ssh(f"docker rm -f {cname} >/dev/null 2>&1; rm -rf {wt} {out}; rmdir /tmp/sdaa_locks/card{st['card']} 2>/dev/null; true")
    os.remove(state_path(tree, a.arm))
    print(f"[pt] DONE arm={a.arm} result_json={'OK' if ok else 'MISSING'} -> runs/{a.arm}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
