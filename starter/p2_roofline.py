#!/usr/bin/env python3
"""
p2_roofline.py -- turn p2_gemm_sweep.py output into a roofline plot.

CSE 40702, The AI Compute Stack.

    python p2_roofline.py --csv gemm.csv --gpu h100_sxm --out roofline.png

THE CEILINGS TABLE BELOW IS DELIBERATELY INCOMPLETE AND PARTLY WRONG.
You must check every number you use against the vendor datasheet and cite the
URL in your report.  Two specific hazards:

  * NVIDIA product pages print tensor-core numbers WITH STRUCTURED SPARSITY
    by default, marked with an asterisk or a footnote.  A dense GEMM cannot
    reach that number.  Dense is exactly half.  If your measured point sits
    at 50% of "peak", check this first.
  * A100 has no FP8.  Turing (Quadro RTX 6000) and Volta (V100) have no BF16,
    no TF32 and no FP8.  Know what your silicon can actually do.

Numbers below are DENSE unless the key says otherwise, and are provided as a
starting point only.  Fix them, cite them, and say in your report which
number you used and where it came from.
"""

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DENSE peak compute (TFLOP/s) and HBM bandwidth (GB/s).  VERIFY THESE.
# ---------------------------------------------------------------------------
GPUS = {
    "a100_40_pcie": dict(bw=1555, fp32=19.5, tf32=156, bf16=312, fp16=312, int8=624),
    "a100_80_sxm":  dict(bw=2039, fp32=19.5, tf32=156, bf16=312, fp16=312, int8=624),
    "h100_sxm":     dict(bw=3350, fp32=67,   tf32=494.5, bf16=989.5, fp16=989.5,
                         fp8=1979, int8=1979),
    "h100_nvl":     dict(bw=3900, fp32=60,   tf32=417.5, bf16=835.5, fp16=835.5,
                         fp8=1670.5, int8=1670.5),
    "h200_sxm":     dict(bw=4800, fp32=67,   tf32=494.5, bf16=989.5, fp16=989.5,
                         fp8=1979, int8=1979),
    "b200":         dict(bw=8000, fp32=None, tf32=1125, bf16=2250, fp16=2250,
                         fp8=4500, int8=4500),
    "mi300x":       dict(bw=5300, fp32=163.4, tf32=653.7, bf16=1300, fp16=1300,
                         fp8=2610, int8=2600),
    "mi355x":       dict(bw=8000, fp32=157.3, bf16=2516.6, fp16=2516.6,
                         fp8=5033.2, int8=5033.2),
}

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load(path):
    with open(path) as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gpu", required=True, choices=list(GPUS),
                    help="which ceilings to draw -- VERIFY them first")
    ap.add_argument("--out", default="roofline.png")
    ap.add_argument("--attention-point", nargs=2, type=float, metavar=("I", "TFLOPS"),
                    help="arithmetic intensity and achieved TFLOP/s of an LLM "
                         "attention layer, computed by you, to overlay")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("empty CSV", file=sys.stderr)
        return 1
    spec = GPUS[args.gpu]
    bw = spec["bw"] / 1e3          # GB/s -> TB/s so units are TFLOP / (TB/s)

    by_prec = defaultdict(list)
    for r in rows:
        by_prec[r["precision"]].append(
            (float(r["arithmetic_intensity"]), float(r["achieved_tflops"]),
             f"{r['M']}x{r['N']}x{r['K']}"))

    fig, ax = plt.subplots(figsize=(9, 6))

    # --- roofs -------------------------------------------------------------
    xs = [2 ** i for i in range(-2, 14)]
    for i, prec in enumerate(sorted(by_prec)):
        peak = spec.get(prec)
        if not peak:
            continue
        ys = [min(peak, x * bw) for x in xs]
        ax.plot(xs, ys, lw=2.5, color=COLORS[i % len(COLORS)], alpha=0.45)
        ridge = peak / bw
        ax.annotate(f"{prec} peak {peak:g} TFLOP/s\nridge I={ridge:.0f}",
                    xy=(xs[-1], peak), xytext=(-4, 4),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=COLORS[i % len(COLORS)])

    # --- measured points ---------------------------------------------------
    for i, prec in enumerate(sorted(by_prec)):
        pts = by_prec[prec]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=55, zorder=5,
                   color=COLORS[i % len(COLORS)], label=prec,
                   edgecolor="white", linewidth=0.8)

    if args.attention_point:
        I, P = args.attention_point
        ax.scatter([I], [P], marker="*", s=420, color="black", zorder=6,
                   label="attention layer")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("arithmetic intensity  (FLOP / byte, cold cache)", fontsize=13)
    ax.set_ylabel("achieved performance  (TFLOP/s)", fontsize=13)
    ax.grid(True, which="both", color="#c8c8c8", lw=0.5, alpha=0.6)
    ax.legend(loc="upper left", frameon=True, edgecolor="#707070", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.text(0.01, 0.005,
             f"source: {args.csv} | ceilings: {args.gpu} (DENSE -- verify against "
             f"datasheet and cite) | bandwidth {spec['bw']} GB/s",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".pdf")
    print(f"wrote {args.out}")

    # --- the classification the memo needs ---------------------------------
    print(f"\n{'shape':>18} {'prec':>5} {'I':>8} {'TFLOP/s':>9} "
          f"{'% of roof':>10}  bound-by")
    for prec in sorted(by_prec):
        peak = spec.get(prec)
        if not peak:
            continue
        ridge = peak / bw
        for I, P, shape in sorted(by_prec[prec]):
            roof = min(peak, I * bw)
            print(f"{shape:>18} {prec:>5} {I:>8.1f} {P:>9.1f} "
                  f"{100*P/roof:>9.0f}%  "
                  f"{'compute' if I >= ridge else 'bandwidth'}")
    print("\nA point far below its roof is neither compute- nor bandwidth-bound. "
          "Say what it IS bound by.  That sentence is worth more than the plot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
