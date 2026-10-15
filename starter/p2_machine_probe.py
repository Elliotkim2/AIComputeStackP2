#!/usr/bin/env python3
"""
p2_machine_probe.py -- Project 2 starter, Task 1: what can this machine
actually do?

CSE 40702, The AI Compute Stack.

Before you can say a kernel reached 43% of peak, you need a defensible number
for "peak."  The datasheet gives you a theoretical ceiling that no real kernel
reaches.  This script measures two ACHIEVABLE ceilings on the GPU in front of
you:

  * achievable peak compute, by sweeping large GEMM shapes and keeping the best
  * achievable HBM bandwidth, by streaming buffers far larger than L2

    python p2_machine_probe.py
    python p2_machine_probe.py --precisions fp32 tf32 bf16 --csv machine.csv

Then compare against the datasheet yourself.  Expect roughly 70-85% of the
DENSE datasheet compute number and 80-90% of the datasheet bandwidth number.
If you get 100% of "peak," you have a measurement error.  If you get 200%, you
halved a sparse number by accident, or forgot it was already sparse.

WHY BOTH NUMBERS MATTER
-----------------------
The roofline has two ceilings and a ridge point at their ratio:

    ridge  I* = peak_FLOPS / peak_bandwidth      [FLOP per byte]

Below I* you are bandwidth-bound; above it, compute-bound.  Using datasheet
numbers gives you the theoretical ridge; using these gives you the achievable
one.  Report both, and say which one you drew.  They differ by enough to move
kernels across the boundary.
"""

import argparse
import csv
import json
import statistics
import sys

import torch

DTYPES = {"fp32": torch.float32, "tf32": torch.float32,
          "bf16": torch.bfloat16, "fp16": torch.float16}


def set_precision(name):
    if name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif name == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def l2_bytes(device):
    try:
        return torch.cuda.get_device_properties(device).L2_cache_size
    except Exception:
        return 50 * 2 ** 20


def time_events(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) / 1e3 for a, b in zip(s, e)]


# --------------------------------------------------------------------------
# achievable compute
# --------------------------------------------------------------------------
def probe_compute(prec, device, shapes, warmup, iters):
    """Largest sustained TFLOP/s over a set of big square-ish GEMMs."""
    dtype = DTYPES[prec]
    set_precision(prec)
    best = None
    for (M, N, K) in shapes:
        try:
            A = torch.randn(M, K, dtype=dtype, device=device)
            B = torch.randn(K, N, dtype=dtype, device=device)
        except torch.cuda.OutOfMemoryError:
            continue
        ts = time_events(lambda: A @ B, warmup, iters)
        med = statistics.median(ts)
        tflops = 2.0 * M * N * K / med / 1e12
        if best is None or tflops > best["achieved_tflops"]:
            best = dict(precision=prec, M=M, N=N, K=K,
                        seconds_median=med, achieved_tflops=tflops)
        del A, B
        torch.cuda.empty_cache()
    return best


# --------------------------------------------------------------------------
# achievable bandwidth
# --------------------------------------------------------------------------
def probe_bandwidth(device, size_mb, warmup, iters):
    """Stream buffers far larger than L2 and report effective GB/s.

    Three kernels, because they move different amounts of traffic and a
    student should see that "bandwidth" is not one number:
      copy  : read 1, write 1  -> 2 bytes moved per element
      scale : read 1, write 1  -> 2
      triad : read 2, write 1  -> 3
    """
    n = (size_mb * 2 ** 20) // 4                     # fp32 elements
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    c = torch.empty_like(a)
    out = []
    for name, fn, traffic in (
            ("copy",  lambda: c.copy_(a),                   2),
            ("scale", lambda: torch.mul(a, 3.0, out=c),     2),
            ("triad", lambda: torch.add(a, b, alpha=3.0, out=c), 3)):
        ts = time_events(fn, warmup, iters)
        med = statistics.median(ts)
        gbs = traffic * n * 4 / med / 1e9
        out.append(dict(kernel=name, buffer_MB=size_mb,
                        seconds_median=med, achieved_GB_s=gbs))
    del a, b, c
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"],
                    choices=list(DTYPES))
    ap.add_argument("--buffer-mb", type=int, default=1024,
                    help="stream buffer size; must be >> L2 (default 1024)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    l2 = l2_bytes(device)

    prov = {"gpu": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_memory_GB": round(props.total_memory / 1e9, 1),
            "l2_cache_MB": round(l2 / 2 ** 20, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda}
    print("PROVENANCE " + json.dumps(prov))

    if args.buffer_mb * 2 ** 20 < 4 * l2:
        print(f"\n  !! buffer is only {args.buffer_mb} MB against an L2 of "
              f"{l2/2**20:.0f} MB. Raise --buffer-mb or your bandwidth number "
              f"will be an L2 number.")

    shapes = [(8192, 8192, 8192), (4096, 4096, 16384),
              (16384, 4096, 4096), (8192, 8192, 4096)]

    rows = []
    print("\n--- achievable compute (best over 4 large GEMM shapes) ---------")
    print(f"{'prec':>6} {'shape':>22} {'TFLOP/s':>10}")
    for prec in args.precisions:
        r = probe_compute(prec, device, shapes, args.warmup, max(10, args.iters // 5))
        if r is None:
            print(f"{prec:>6}  all shapes OOM, skipped")
            continue
        r.update(prov)
        rows.append({"probe": "compute", **r})
        print(f"{prec:>6} {r['M']}x{r['N']}x{r['K']:>6} {r['achieved_tflops']:>10.1f}")

    print("\n--- achievable HBM bandwidth ----------------------------------")
    print(f"{'kernel':>8} {'GB/s':>10}")
    for r in probe_bandwidth(device, args.buffer_mb, args.warmup, args.iters):
        r.update(prov)
        rows.append({"probe": "bandwidth", **r})
        print(f"{r['kernel']:>8} {r['achieved_GB_s']:>10.0f}")

    comp = [r for r in rows if r["probe"] == "compute"]
    bw = max((r["achieved_GB_s"] for r in rows if r["probe"] == "bandwidth"),
             default=0)
    if comp and bw:
        print("\n--- achievable ridge points -----------------------------------")
        print(f"{'prec':>6} {'peak TFLOP/s':>13} {'BW GB/s':>9} "
              f"{'ridge FLOP/byte':>16}")
        for r in comp:
            print(f"{r['precision']:>6} {r['achieved_tflops']:>13.1f} "
                  f"{bw:>9.0f} {r['achieved_tflops']*1e12/(bw*1e9):>16.1f}")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    print("\nNow, for the report:")
    print("  1. What fraction of the DENSE datasheet number did you reach, "
          "for compute and for bandwidth separately? Cite the datasheet.")
    print("  2. Which ridge point will you draw, datasheet or achievable, "
          "and which of your kernels change category between them?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
