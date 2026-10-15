#!/usr/bin/env python3
"""
p2_gemm_sweep.py -- Project 2 starter, "Roofline Analysis".

CSE 40702, The AI Compute Stack.

Times matrix multiplies across shapes and precisions, correctly, and writes a
CSV that p2_roofline.py turns into a roofline plot.

    python p2_gemm_sweep.py --precisions fp32 tf32 bf16 --shapes square --csv gemm.csv
    python p2_roofline.py --csv gemm.csv --gpu h100_sxm --out roofline.png

WHY YOUR FIRST ATTEMPT WILL BE WRONG
------------------------------------
Almost every naive GPU benchmark is wrong in one of five ways.  This script
handles all five; if you write your own, you must handle them too, and the
oral will ask you which ones you hit.

 1. NO SYNCHRONIZATION.  CUDA kernel launches are asynchronous.  time.time()
    around a matmul measures how long it took to QUEUE the work, not to do
    it.  Fix: torch.cuda.Event, or torch.cuda.synchronize() around the timed
    region.
 2. NO WARMUP.  cuBLAS autotunes on first call and caches the choice in the
    handle; the GPU also boots from a low clock state.  The first iteration
    can be 10x slow.  Fix: >= 10 warmup iterations, and NVIDIA's own CUTLASS
    guidance is to warm until POWER stabilizes, roughly 3 seconds.
 3. HOT L2.  Reusing the same A and B buffers every iteration means the
    second iteration reads from L2, not HBM.  Small shapes then look
    compute-bound when they are not.  Fix: rotate over a pool of buffers
    totalling at least 2x L2 capacity.  This script does that.
 4. TF32 MASQUERADING AS FP32.  On Ampere and later, PyTorch may run an
    "fp32" matmul on TF32 tensor cores, which is a ~8x different ceiling.
    Fix: set the precision explicitly.  This script does, and records which.
 5. MEAN INSTEAD OF MEDIAN.  One thermal excursion ruins a mean.  Report the
    median and also report the spread.

WHAT THE ROOFLINE NEEDS FROM YOU
--------------------------------
For a GEMM of shape (M, N, K):
    FLOPs  = 2 * M * N * K                      (one multiply + one add)
    Bytes  = (M*K + K*N + M*N) * bytes_per_elem (cold-cache, read A, read B,
                                                 write C)
    Arithmetic intensity  I = FLOPs / Bytes     (FLOP per byte)
    Achieved performance  P = FLOPs / seconds
The roofline ceiling is  P <= min(peak_flops, I * peak_bandwidth).

Do NOT use torch.profiler's with_flops.  Compute 2*M*N*K yourself.  It is
exact and it is the point of the exercise.
"""

import argparse
import csv
import json
import statistics
import sys

import torch

DTYPES = {
    "fp32": torch.float32,
    "tf32": torch.float32,      # same storage, different tensor-core path
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

SHAPE_SETS = {
    "square": [(n, n, n) for n in (256, 512, 1024, 2048, 4096, 8192)],
    "skinny": [(4096, 4096, k) for k in (64, 128, 256, 512, 1024, 2048, 4096)],
    "tall":   [(m, 4096, 4096) for m in (128, 256, 512, 1024, 4096, 16384)],
    "batch":  [(b * 512, 4096, 4096) for b in (1, 4, 16, 64)],
}


def set_precision(name):
    """Return a human-readable description of the fp32 matmul path in force."""
    if name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return "fp32 storage, TF32 tensor cores"
    if name == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
        return "fp32 storage, true FP32 CUDA cores"
    return f"{name} storage, {name} tensor cores"


def l2_bytes(device):
    try:
        return torch.cuda.get_device_properties(device).L2_cache_size
    except Exception:
        return 50 * 2 ** 20        # assume 50 MB if the attribute is absent


def make_pool(M, N, K, dtype, device, target_bytes):
    """Allocate enough (A, B) pairs to exceed target_bytes so L2 goes cold."""
    per_pair = (M * K + K * N) * torch.tensor([], dtype=dtype).element_size()
    n = max(2, min(16, int(target_bytes // max(per_pair, 1)) + 1))
    pool = []
    for _ in range(n):
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        pool.append((A, B))
    return pool


def time_gemm(M, N, K, prec, device, warmup=20, iters=100):
    dtype = DTYPES[prec]
    set_precision(prec)
    try:
        pool = make_pool(M, N, K, dtype, device, 2 * l2_bytes(device))
    except torch.cuda.OutOfMemoryError:
        return None

    A, B = pool[0]
    for i in range(warmup):
        A, B = pool[i % len(pool)]
        C = A @ B
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        A, B = pool[i % len(pool)]
        starts[i].record()
        C = A @ B
        ends[i].record()
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    del pool, C
    torch.cuda.empty_cache()

    med = statistics.median(ms) / 1e3
    elem = torch.tensor([], dtype=dtype).element_size()
    flops = 2.0 * M * N * K
    byts = (M * K + K * N + M * N) * elem
    return dict(
        M=M, N=N, K=K, precision=prec,
        seconds_median=med,
        seconds_min=min(ms) / 1e3,
        seconds_p90=sorted(ms)[int(0.9 * len(ms))] / 1e3,
        spread_pct=100.0 * (max(ms) - min(ms)) / statistics.median(ms),
        flops=flops,
        bytes_cold=byts,
        arithmetic_intensity=flops / byts,
        achieved_tflops=flops / med / 1e12,
        achieved_gbytes_per_s=byts / med / 1e9,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"],
                    choices=list(DTYPES))
    ap.add_argument("--shapes", nargs="+", default=["square"],
                    choices=list(SHAPE_SETS))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--csv", default="gemm.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("This benchmark needs a CUDA GPU.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    prov = {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_GB": round(props.total_memory / 1e9, 1),
        "l2_cache_MB": round(l2_bytes(device) / 2 ** 20, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    # PROVENANCE LINE.  This must appear in your report, verbatim, next to
    # the ceilings you used and whether those ceilings are DENSE or SPARSE.
    print("PROVENANCE " + json.dumps(prov))

    shapes = []
    for s in args.shapes:
        shapes += SHAPE_SETS[s]

    rows = []
    print(f"\n{'M':>6}{'N':>6}{'K':>6} {'prec':>5} {'I (F/B)':>9} "
          f"{'TFLOP/s':>9} {'GB/s':>8} {'spread':>7}")
    for (M, N, K) in shapes:
        for prec in args.precisions:
            r = time_gemm(M, N, K, prec, device, args.warmup, args.iters)
            if r is None:
                print(f"{M:>6}{N:>6}{K:>6} {prec:>5}   OOM, skipped")
                continue
            r.update(prov)
            rows.append(r)
            print(f"{M:>6}{N:>6}{K:>6} {prec:>5} {r['arithmetic_intensity']:>9.1f} "
                  f"{r['achieved_tflops']:>9.1f} {r['achieved_gbytes_per_s']:>8.0f} "
                  f"{r['spread_pct']:>6.1f}%")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv} ({len(rows)} rows)")
    print("Next: python p2_roofline.py --csv %s --gpu <your gpu>" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
