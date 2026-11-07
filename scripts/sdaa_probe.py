#!/usr/bin/env python3
"""SDAA multi-card capability probe for TensorLBM."""
import json, os, sys, time
import torch

WORK_DIR = os.environ.get("WORK_DIR", "/root/tp-agent/work/m4991")

def probe_device_count():
    return torch.sdaa.device_count()

def probe_backends():
    result = {}
    for b in ['nccl', 'gloo', 'mpi', 'ucc']:
        result[b] = torch.distributed.is_available() and torch.distributed.is_backend_available(b)
    return result

def probe_device_bandwidth(sizes_mb=None):
    if sizes_mb is None:
        sizes_mb = [1, 4, 16, 64]
    d = torch.device('sdaa:0')
    results = {}
    for sz_mb in sizes_mb:
        n = sz_mb * 1024 * 1024 // 4
        x_cpu = torch.randn(n, device='cpu')
        y_sdaa = torch.empty(n, device=d)
        torch.sdaa.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y_sdaa.copy_(x_cpu)
        torch.sdaa.synchronize()
        t1 = time.perf_counter()
        h2d = sz_mb * 10 / (t1 - t0)
        t0 = time.perf_counter()
        for _ in range(10):
            x_cpu.copy_(y_sdaa)
        torch.sdaa.synchronize()
        t1 = time.perf_counter()
        d2h = sz_mb * 10 / (t1 - t0)
        results[f"{sz_mb}MB"] = {"h2d_mbps": round(h2d, 1), "d2h_mbps": round(d2h, 1)}
    return results

def probe_gloo_allreduce_bandwidth(sizes_mb=None):
    if sizes_mb is None:
        sizes_mb = [1, 4, 16, 64]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    results = {}
    for sz_mb in sizes_mb:
        n = sz_mb * 1024 * 1024 // 4
        tensor = torch.randn(n, device='cpu')
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()
        t0 = time.perf_counter()
        for _ in range(10):
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.barrier()
        t1 = time.perf_counter()
        if rank == 0:
            elapsed = t1 - t0
            bw = sz_mb * 10 * 2 / elapsed
            results[f"{sz_mb}MB"] = round(bw, 1)
    return results

def probe_correctness():
    """2-rank allreduce SUM correctness: each rank fills with its rank value,
    after SUM all ranks should have sum(rank_0, rank_1, ..., rank_{N-1})."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        return False
    n = 1024
    tensor = torch.full((n,), float(rank), device='cpu')
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # Expected: sum of 0+1+...+(world_size-1) = world_size*(world_size-1)/2
    expected = float(world_size * (world_size - 1) // 2)
    correct = torch.allclose(tensor, torch.full((n,), expected))
    return bool(correct)

def main():
    mode = os.environ.get("PROBE_MODE", "single")

    if mode == "single":
        device_count = probe_device_count()
        backends = probe_backends()
        dev_bw = probe_device_bandwidth()
        result = {
            "device_count": device_count,
            "backends": backends,
            "device_bandwidth_mbps": dev_bw,
            "gloo_bandwidth_mbps": {},
            "correctness_pass": None
        }
        outpath = os.path.join(WORK_DIR, "artifacts/p8/sdaa_probe.json")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[single] device_count={device_count}, backends={backends}")
        print(f"[single] device_bandwidth={dev_bw}")
        print(f"[single] wrote {outpath}")

    elif mode == "multi":
        torch.distributed.init_process_group(backend="gloo")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"[multi] rank={rank}/{world_size} initialized")
        gloo_bw = probe_gloo_allreduce_bandwidth()
        if rank == 0:
            outpath = os.path.join(WORK_DIR, "artifacts/p8/sdaa_probe_gloo.json")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(outpath, "w") as f:
                json.dump(gloo_bw, f, indent=2)
            print(f"[multi] gloo_bandwidth={gloo_bw}")
        torch.distributed.destroy_process_group()

    elif mode == "correctness":
        torch.distributed.init_process_group(backend="gloo")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"[correctness] rank={rank}/{world_size} initialized")
        ok = probe_correctness()
        if rank == 0:
            outpath = os.path.join(WORK_DIR, "artifacts/p8/sdaa_probe_correctness.json")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            with open(outpath, "w") as f:
                json.dump({"correctness_pass": ok}, f, indent=2)
            print(f"[correctness] pass={ok}")
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
