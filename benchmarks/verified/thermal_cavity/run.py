"""W2-A 基准入口：de Vahl Davis 热腔六案（Ra=1e3/1e4 × N=64/128/256）。

物理全部在 tensorlbm 共性模块（driver.py 只编排）；本文件零本地物理实现。
稳态判据（NOTES.md 预注册）：Nu 漂移 <1e-4/1000 步（nu/nu_left/nu_right 三者
取 max，1000 步窗），连续 3 窗触发；触发后再跑 max(10000, 10%·t_stop) 验证段。

用法：
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=<src_patched>:<本目录> .../python run.py --cases all
  .../python run.py --timing   # 先测各档步耗时
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import driver

MAX_STEPS = {
    1e3: {64: 300_000, 128: 700_000, 256: 1_600_000},
    1e4: {64: 200_000, 128: 450_000, 256: 1_000_000},
}
CASES = [(ra, n) for ra in (1e3, 1e4) for n in (64, 128, 256)]


def tag_of(ra: float, nx: int) -> str:
    return f"ra1e{int(round(__import__('math').log10(ra)))}_n{nx}"


def run_case(
    ra: float,
    nx: int,
    device: torch.device,
    log_dir: str,
    out_dir: str,
    min_steps: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """min_steps：触发许可下限（0=无）。v6 首轮 n256 两案在初始发展段
    假触发（u_max 仍在爬升、Nu 冻结在导热值附近使 drift<1e-4 提前满足，
    验证段判决 post_move/last_drift 超阈暴露），据此为 N≥128 补设
    floor = 1.15·t_stop(N=64)·(H/63)²（NOTES.md 披露；门限本身不变）。
    """
    import os

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    tag = tag_of(ra, nx)
    t_hot, t_cold = 1.0, 0.0
    f, g, wall, p = driver.make_fields(nx, ra, 0.71, 0.6, device, dtype=dtype)
    tau_T, g_beta = p["tau_T"], p["g_beta"]

    max_steps = MAX_STEPS[ra][nx]
    every = 1000
    log_path = f"{log_dir}/case_{tag}.log"
    log = open(log_path, "w", buffering=1)
    import tensorlbm.thermal as th_mod

    th_sha = hashlib.sha256(open(th_mod.__file__, "rb").read()).hexdigest()[:12]
    log.write(
        f"# case {tag} thermal={th_mod.__file__} sha12={th_sha}\n"
        f"# params {json.dumps(p)} tau=0.6 dtype={dtype} device={device}\n"
        "# step nu_left nu_right nu u_max_nd v_max_nd drift\n"
    )

    prev = None
    quiet = 0
    t_stop = None
    target = max_steps
    hist = []
    t0 = time.time()
    step = 0
    while step < target:
        step += 1
        f, g = driver.advance(f, g, wall, 0.6, tau_T, g_beta, t_hot, t_cold)
        if step % every == 0:
            obs = driver.observables(f, g, p, t_hot, t_cold)
            drift = (
                0.0
                if prev is None
                else max(abs(obs[k] - prev[k]) for k in ("nu", "nu_left", "nu_right"))
            )
            log.write(
                f"{step} {obs['nu_left']:.8f} {obs['nu_right']:.8f} {obs['nu']:.8f} "
                f"{obs['u_max_nd']:.6f} {obs['v_max_nd']:.6f} {drift:.3e}\n"
            )
            hist.append({"step": step, "nu": obs["nu"], "drift": drift})
            prev = obs
            if t_stop is None:
                if prev is not None and step > every and step >= min_steps and drift < 1e-4:
                    quiet += 1
                else:
                    quiet = 0
                if quiet >= 3:
                    t_stop = step
                    target = min(max_steps, step + max(10_000, step // 10))
                    log.write(f"# steady trigger at {step}; verification segment to {target}\n")
    elapsed = time.time() - t0

    obs = driver.observables(f, g, p, t_hot, t_cold)
    nu_at_stop = (
        next((h["nu"] for h in reversed(hist) if h["step"] <= t_stop), None) if t_stop else None
    )
    result = {
        "tag": tag,
        "ra": ra,
        "pr": 0.71,
        "nx": nx,
        "ny": nx,
        "tau": 0.6,
        "params": p,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "thermal_file": th_mod.__file__,
        "thermal_sha12": th_sha,
        "steps_run": step,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "steady_triggered": t_stop is not None,
        "t_stop": t_stop,
        "verification_steps": step - (t_stop or step),
        "post_trigger_nu_move": None if nu_at_stop is None else obs["nu"] - nu_at_stop,
        "last_drift": hist[-1]["drift"] if hist else None,
        "elapsed_s": round(elapsed, 1),
        "obs": obs,
    }
    with open(f"{out_dir}/case_{tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    log.write(f"# final {json.dumps(obs)}\n")
    log.close()
    print(
        f"[run] {tag}: steps={step} steady={t_stop} nu={obs['nu']:.6f} "
        f"u_max={obs['u_max_nd']:.4f} v_max={obs['v_max_nd']:.4f} "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    return result


def timing(device: torch.device) -> None:
    for nx in (64, 128, 256):
        f, g, wall, p = driver.make_fields(nx, 1e4, 0.71, 0.6, device)
        for _ in range(50):
            f, g = driver.advance(f, g, wall, 0.6, p["tau_T"], p["g_beta"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            f, g = driver.advance(f, g, wall, 0.6, p["tau_T"], p["g_beta"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 200
        print(f"[timing] N={nx}: {dt * 1e3:.2f} ms/step", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all", help="逗号分隔 tag（ra1e3_n64…）或 all")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-floor", action="store_true", help="禁用触发许可下限（复现 v6 首轮）")
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="ra1e3_n256 需 float64：浮力步进 <1 ULP(fp32)（NOTES.md fp32 量化伪影）",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.timing:
        timing(device)
        return
    import os

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    todo = (
        CASES
        if args.cases == "all"
        else [(float(t.split("_")[0][2:]), int(t.split("_")[1][1:])) for t in args.cases.split(",")]
    )
    print(
        f"[run] device={device} torch={torch.__version__} cases={[tag_of(r, n) for r, n in todo]}",
        flush=True,
    )
    # 触发许可下限（v6a 修订，见 run_case docstring）：floor = 1.15·t_stop(n64,v6)·(H/63)²
    FLOORS = {
        "ra1e3_n128": 127_000,
        "ra1e3_n256": 510_000,
        "ra1e4_n128": 103_000,
        "ra1e4_n256": 415_000,
    }
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    for ra, nx in todo:
        tag = tag_of(ra, nx)
        floor = 0 if args.no_floor else FLOORS.get(tag, 0)
        run_case(ra, nx, device, args.log_dir, args.out_dir, min_steps=floor, dtype=dtype)


if __name__ == "__main__":
    main()
