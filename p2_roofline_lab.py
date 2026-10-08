#!/usr/bin/env python3
"""
p2_roofline_lab.py -- Project 2 starter: build a roofline for YOUR machine.

CSE 40702, The AI Compute Stack.

RUNS ON A LAPTOP.  NumPy only.  No GPU, no cluster account, no PyTorch.
A full run is two to five minutes.

    pip install numpy matplotlib
    python p2_roofline_lab.py --csv p2_group07
    python p2_roofline_lab.py --machine-only        # just the two ceilings

WHAT IT DOES
------------
  1. Measures the two ceilings of the machine you are sitting at:
     achievable compute (GFLOP/s) and achievable DRAM bandwidth (GB/s).
  2. Sweeps matrix multiplies across shapes so that arithmetic intensity
     traverses the ridge point, and times each one correctly.
  3. Adds two anchor kernels with known, very low intensity, so you can see
     the bandwidth roof from below.
  4. Writes a CSV and, with --plot, a roofline figure.

WHY A LAPTOP ROOFLINE IS NOT A CONSOLATION PRIZE
------------------------------------------------
The roofline model does not know what kind of processor it is describing.  A
laptop CPU has a peak FLOP rate, a DRAM bandwidth, and therefore a ridge point,
in exactly the way an H100 does.  The numbers differ by two or three orders of
magnitude; the shape of the argument does not.

You will measure your own ridge point, then place datasheet ceilings for real
accelerators on the same axes and compare.  A machine whose ridge is at 30
FLOP/byte and one whose ridge is at 295 are asking for different code, and
seeing that on one plot is the point of the project.

FIVE WAYS TO TIME THIS WRONG
----------------------------
 1. No warmup.  The first call allocates, pages in memory, and lets BLAS pick
    a kernel.  It can be several times slow.
 2. Mean instead of median.  One scheduling hiccup ruins a mean.
 3. Hot cache.  Re-using the same small arrays means you are measuring L2 or
    L3, not DRAM.  Anything intended to measure bandwidth must use buffers far
    larger than the last-level cache.
 4. Counting the wrong FLOPs.  For a GEMM it is exactly 2*M*N*K.  Compute it
    analytically; do not trust a library to tell you.
 5. Forgetting that BLAS is multi-threaded.  Your GFLOP/s number is for all
    cores at once.  Say how many.  Set OMP_NUM_THREADS if you want one.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

DTYPES = {"fp32": np.float32, "fp64": np.float64}


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


# --------------------------------------------------------------------------
# ceiling 1: achievable compute
# --------------------------------------------------------------------------
def probe_compute(dtype_name, sizes, warmup, iters):
    dt = DTYPES[dtype_name]
    best = None
    rows = []
    for n in sizes:
        A = np.random.rand(n, n).astype(dt)
        B = np.random.rand(n, n).astype(dt)
        med, lo, hi = timeit(lambda: A @ B, warmup, iters)
        g = 2.0 * n ** 3 / med / 1e9
        rows.append(dict(probe="compute", dtype=dtype_name, n=n,
                         seconds_median=med, gflops=g,
                         spread_pct=100 * (hi - lo) / med))
        if best is None or g > best:
            best = g
        del A, B
    return best, rows


# --------------------------------------------------------------------------
# ceiling 2: achievable DRAM bandwidth
# --------------------------------------------------------------------------
def probe_bandwidth(dtype_name, buffer_mb, warmup, iters):
    """copy, scale and triad on buffers far larger than last-level cache.

    traffic counts bytes that must cross the memory bus:
      copy  c = a          read 1 + write 1 = 2
      scale c = 3a         read 1 + write 1 = 2
      triad c = a + 3b     read 2 + write 1 = 3
    """
    dt = DTYPES[dtype_name]
    elem = np.dtype(dt).itemsize
    n = (buffer_mb * 2 ** 20) // elem
    a = np.random.rand(n).astype(dt)
    b = np.random.rand(n).astype(dt)
    c = np.empty_like(a)
    rows = []
    for name, fn, traffic in (
            ("copy",  lambda: np.copyto(c, a), 2),
            ("scale", lambda: np.multiply(a, dt(3.0), out=c), 2),
            ("triad", lambda: np.add(a, dt(3.0) * b, out=c), 3)):
        med, lo, hi = timeit(fn, warmup, iters)
        rows.append(dict(probe="bandwidth", dtype=dtype_name, kernel=name,
                         buffer_MB=buffer_mb, seconds_median=med,
                         gbytes_per_s=traffic * n * elem / med / 1e9,
                         spread_pct=100 * (hi - lo) / med))
    best = max(r["gbytes_per_s"] for r in rows)
    del a, b, c
    return best, rows


# --------------------------------------------------------------------------
# the sweep: shapes chosen so intensity crosses the ridge
# --------------------------------------------------------------------------
def gemm_point(M, N, K, dtype_name, warmup, iters):
    dt = DTYPES[dtype_name]
    elem = np.dtype(dt).itemsize
    A = np.random.rand(M, K).astype(dt)
    B = np.random.rand(K, N).astype(dt)
    med, lo, hi = timeit(lambda: A @ B, warmup, iters)
    flops = 2.0 * M * N * K
    byts = (M * K + K * N + M * N) * elem
    del A, B
    return dict(probe="gemm", dtype=dtype_name, M=M, N=N, K=K,
                seconds_median=med,
                flops=flops, bytes_cold=byts,
                arithmetic_intensity=flops / byts,
                gflops=flops / med / 1e9,
                gbytes_per_s=byts / med / 1e9,
                spread_pct=100 * (hi - lo) / med)


def anchor_points(dtype_name, buffer_mb, warmup, iters):
    """Two kernels with tiny, exactly-known intensity, to pin the bandwidth
    roof from below.  axpy does 2 FLOP per 12 bytes at fp32; sum does 1 FLOP
    per 4 bytes."""
    dt = DTYPES[dtype_name]
    elem = np.dtype(dt).itemsize
    n = (buffer_mb * 2 ** 20) // elem
    a = np.random.rand(n).astype(dt)
    b = np.random.rand(n).astype(dt)
    c = np.empty_like(a)
    out = []
    med, lo, hi = timeit(lambda: np.add(a, dt(3.0) * b, out=c), warmup, iters)
    out.append(dict(probe="anchor", dtype=dtype_name, kernel="axpy", n=n,
                    seconds_median=med, flops=2.0 * n,
                    bytes_cold=3.0 * n * elem,
                    arithmetic_intensity=2.0 / (3 * elem),
                    gflops=2.0 * n / med / 1e9,
                    gbytes_per_s=3.0 * n * elem / med / 1e9,
                    spread_pct=100 * (hi - lo) / med))
    med, lo, hi = timeit(lambda: a.sum(), warmup, iters)
    out.append(dict(probe="anchor", dtype=dtype_name, kernel="sum", n=n,
                    seconds_median=med, flops=1.0 * n,
                    bytes_cold=1.0 * n * elem,
                    arithmetic_intensity=1.0 / elem,
                    gflops=1.0 * n / med / 1e9,
                    gbytes_per_s=1.0 * n * elem / med / 1e9,
                    spread_pct=100 * (hi - lo) / med))
    del a, b, c
    return out


SHAPE_FAMILIES = {
    # sweeping K at fixed M,N walks intensity from far below the ridge to far
    # above it.  This is the family that actually shows the corner on a CPU.
    "skinny": lambda: [(1024, 1024, k) for k in
                       (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)],
    "square": lambda: [(n, n, n) for n in
                       (32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536)],
    "tall":   lambda: [(m, 512, 512) for m in
                       (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--family", default="skinny", choices=list(SHAPE_FAMILIES))
    ap.add_argument("--buffer-mb", type=int, default=128,
                    help="stream buffer size; must be >> last-level cache")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--machine-only", action="store_true")
    ap.add_argument("--csv", help="prefix for CSV output")
    ap.add_argument("--plot", help="write a roofline PNG to this path")
    args = ap.parse_args()

    prov = {
        "machine": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "dtype": args.dtype,
    }
    try:
        cfg = np.show_config(mode="dicts")
        prov["blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:
        prov["blas"] = "unknown"
    # PROVENANCE.  Paste this into your report verbatim.
    print("PROVENANCE " + json.dumps(prov))
    print("  NOTE: BLAS is multi-threaded. The compute ceiling below is for "
          f"all {os.cpu_count()} logical cores at once. Say so in the report, "
          "or set OMP_NUM_THREADS=1 and measure one core.")

    rows = []

    print("\n--- ceiling 1: achievable compute -----------------------------")
    peak, crows = probe_compute(args.dtype, (256, 512, 1024, 2048),
                                args.warmup, max(3, args.iters // 3))
    rows += crows
    print(f"{'n':>6} {'ms':>9} {'GFLOP/s':>10} {'spread':>8}")
    for r in crows:
        print(f"{r['n']:>6} {r['seconds_median']*1e3:>9.2f} "
              f"{r['gflops']:>10.1f} {r['spread_pct']:>7.1f}%")
    print(f"  achievable compute peak: {peak:.1f} GFLOP/s")

    print("\n--- ceiling 2: achievable DRAM bandwidth ----------------------")
    bw, brows = probe_bandwidth(args.dtype, args.buffer_mb,
                                args.warmup, args.iters)
    rows += brows
    print(f"{'kernel':>8} {'ms':>9} {'GB/s':>9} {'spread':>8}")
    for r in brows:
        print(f"{r['kernel']:>8} {r['seconds_median']*1e3:>9.2f} "
              f"{r['gbytes_per_s']:>9.1f} {r['spread_pct']:>7.1f}%")
    print(f"  achievable bandwidth: {bw:.1f} GB/s")

    ridge = peak / bw
    elem = np.dtype(DTYPES[args.dtype]).itemsize
    print(f"\n  RIDGE POINT  I* = {peak:.1f} GFLOP/s / {bw:.1f} GB/s "
          f"= {ridge:.1f} FLOP/byte")
    print(f"  A square GEMM has I = 2n/(3s) = n/{3*elem//2}, so it crosses the "
          f"ridge near n = {3*elem*ridge/2:.0f}.")

    if args.machine_only:
        return finish(rows, args, prov, ridge, peak, bw)

    print(f"\n--- the sweep: {args.family} family ---------------------------")
    pts = [gemm_point(M, N, K, args.dtype, args.warmup, args.iters)
           for (M, N, K) in SHAPE_FAMILIES[args.family]()]
    pts += anchor_points(args.dtype, args.buffer_mb, args.warmup, args.iters)
    rows += pts
    print(f"{'shape':>18} {'I (F/B)':>9} {'GFLOP/s':>9} {'roof':>9} "
          f"{'% roof':>7}  bound-by")
    for r in pts:
        label = (f"{r['M']}x{r['N']}x{r['K']}" if r["probe"] == "gemm"
                 else r["kernel"])
        I = r["arithmetic_intensity"]
        roof = min(peak, I * bw)
        r["roof_gflops"] = roof
        r["pct_of_roof"] = 100 * r["gflops"] / roof
        r["bound_by"] = "compute" if I >= ridge else "bandwidth"
        print(f"{label:>18} {I:>9.2f} {r['gflops']:>9.1f} {roof:>9.1f} "
              f"{r['pct_of_roof']:>6.0f}%  {r['bound_by']}")

    if args.plot:
        make_plot(pts, peak, bw, ridge, args, prov)

    print("\nFor the report:")
    print("  1. Where is your ridge point, and which of your shapes crossed it?")
    print("  2. Which point is furthest below its roof? What bounds it?")
    print("  3. Put a datasheet accelerator on the same axes. How far does the")
    print("     ridge move, and what does that change about what code you would")
    print("     write for it?")
    return finish(rows, args, prov, ridge, peak, bw)


def make_plot(pts, peak, bw, ridge, args, prov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [2.0 ** i for i in range(-5, 12)]
    ys = [min(peak, x * bw) for x in xs]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs, ys, lw=2.5, color="#377eb8", alpha=0.6, label="roofline")
    ax.axvline(ridge, ls="--", lw=1.2, color="#707070")
    ax.annotate(f"ridge  I* = {ridge:.1f}", xy=(ridge, peak), xytext=(6, -14),
                textcoords="offset points", fontsize=10, color="#404040")
    g = [p for p in pts if p["probe"] == "gemm"]
    a = [p for p in pts if p["probe"] == "anchor"]
    ax.scatter([p["arithmetic_intensity"] for p in g], [p["gflops"] for p in g],
               s=55, color="#e41a1c", zorder=5, edgecolor="white", lw=0.8,
               label="GEMM sweep")
    ax.scatter([p["arithmetic_intensity"] for p in a], [p["gflops"] for p in a],
               s=110, marker="*", color="#4daf4a", zorder=6,
               edgecolor="white", lw=0.8, label="streaming anchors")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("arithmetic intensity  (FLOP / byte, cold cache)", fontsize=13)
    ax.set_ylabel("achieved performance  (GFLOP/s)", fontsize=13)
    ax.grid(True, which="both", color="#c8c8c8", lw=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=True, edgecolor="#707070", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.text(0.01, 0.005,
             f"source: p2_roofline_lab.py | {prov['machine']} | "
             f"{prov['cpu_count']} logical cores | {args.dtype} | "
             f"peak {peak:.0f} GFLOP/s, BW {bw:.1f} GB/s",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.plot, dpi=200)
    print(f"wrote {args.plot}")


def finish(rows, args, prov, ridge, peak, bw):
    if args.csv:
        for r in rows:
            r.update(prov)
            r["ridge_flop_per_byte"] = ridge
            r["machine_peak_gflops"] = peak
            r["machine_bandwidth_gbs"] = bw
        keys = sorted({k for r in rows for k in r})
        path = f"{args.csv}_roofline.csv"
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
